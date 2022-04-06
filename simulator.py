import numpy as np
import pygame
import numpy as np
import time
from render import GolfRender
import agents as agents


class GolfSimulator:
    def __init__(self):
        # Episodes counter
        self.iterations = 0

        # Game score
        self.score = 0

        # Power of a kick (length of the kick)
        self.power = 5

        # Number of possible orientations
        self.orientations = 32

        # Field dimensions (meters)
        self.field_size = np.array([10, 5])

        # Size and position of the goal (hole)
        self.goal_radius = 0.4
        self.goal_position = np.array([3.5, 0])

        # Current position of the ball
        self.ball = np.array([0., 0.])

        # Frames for the ball animation
        self.ball_frames = []

        # Position of obstacles ((x, y), (width, height))
        self.obstacles = [
            # Walls around the field
            [np.array([0, self.field_size[1]/2 + 5]),
             np.array([self.field_size[0]*2, 10])],
            [np.array([0, -self.field_size[1]/2 - 5]),
             np.array([self.field_size[0]*2, 10])],
            [np.array([self.field_size[0]/2 + 5, 0]),
             np.array([10, self.field_size[1]*2])],
            [np.array([-self.field_size[0]/2 - 5, 0]),
             np.array([10, self.field_size[1]*2])],

            [np.array([0, 0]), np.array([1, 1])],
            [np.array([-2, 2]), np.array([1, 1])],
            [np.array([1, -3]), np.array([3.75, 2])],
            [np.array([3, -2]), np.array([1, 1.5])],
            [np.array([1, 2]), np.array([0.6, 3])],
            [np.array([0.5, 0.5]), np.array([0.5, 0.5])],
            [np.array([-2.6, -1]), np.array([0.5, 3.3])],
            [np.array([2.5, -0.5]), np.array([0.5, 3.3])],
            [np.array([3.25, 1]), np.array([1.5, 0.3])]
        ]

    def get_kick_target(self, orientation):
        """Get the current kick target, given an orientation"""
        angle = orientation * 2 * np.pi / self.orientations
        length = self.power

        return np.array([
            np.cos(angle)*length,
            np.sin(angle)*length,
        ])

    def kick(self, orientation, animate=False):
        """Kicks the ball, returns true if the ball reached the hole"""
        self.score += 1

        goal = False
        target = self.get_kick_target(orientation)

        ball = self.ball.copy()
        unit = target / np.linalg.norm(target)
        length = np.linalg.norm(target)
        ball_frames = []

        dl = 0.1
        steps = int(length/dl)

        # Simulating the ball motion
        for step in range(steps):
            obstacle = self.in_obstacle(ball+unit*dl)
            if obstacle:
                hit = self.hit_test(ball, ball+unit*dl, obstacle)

                if hit is not None:
                    if hit == 'x':
                        unit[0] *= -1
                    elif hit == 'y':
                        unit[1] *= -1
            else:
                ball += unit*dl

            ball_frames.append(ball.copy())

            if step > steps*2/3 and self.is_success(ball):
                ball_frames.append(True)
                goal = True
                break

        # Decimating animation frames in the begining
        N = len(ball_frames)
        for k in range(N):
            if k % 2 == 0 or k > 8*N/10:
                self.ball_frames.append(ball_frames[k])

        if not animate:
            self.ball_frames = []
            self.ball = ball

        return goal

    def is_success(self, point):
        """Is a given point a success (in the golf hole) point?"""
        return np.linalg.norm(point - self.goal_position) < self.goal_radius

    def is_valid_starting_position(self, pos):
        """Is the given position a valid starting point (not in obstacle and not success)?"""
        return not self.in_obstacle(pos) and not self.is_success(pos)

    def reset_ball(self, animate=False):
        """Place the ball somewhere random on the field"""
        self.score = 0
        def get_random_position(): return (np.random.rand(
            2) * self.field_size) - self.field_size/2.

        self.ball = get_random_position()
        while not self.is_valid_starting_position(self.ball):
            self.ball = get_random_position()

        if not animate:
            self.ball_frames = []
        else:
            self.ball_frames.append(self.ball.copy())

    def intersection(self, ln1, ln2):
        """Intersection between two given lines"""
        A = np.array(ln1[0])
        u = np.array(ln1[1]) - A
        B = np.array(ln2[0])
        v = np.array(ln2[1]) - B

        try:
            l = np.linalg.inv(np.stack((u, -v)).T).dot(B - A)
            return (l >= 0).all() and (l <= 1).all()
        except np.linalg.LinAlgError:
            return False

    def obstacle_lines(self, obstacle):
        """Get all the lines of an obstacle"""
        x, y, w, h = np.array(obstacle).flatten()

        return [
            ['y', [[x-w/2, y+h/2], [x+w/2, y+h/2]]],
            ['y', [[x-w/2, y-h/2], [x+w/2, y-h/2]]],
            ['x', [[x-w/2, y+h/2], [x-w/2, y-h/2]]],
            ['x', [[x+w/2, y+h/2], [x+w/2, y-h/2]]]
        ]

    def in_obstacle(self, point):
        """Is a given point in an obstacle? (fast test)"""
        for obstacle in self.obstacles:
            tmp = point - obstacle[0]
            if abs(tmp[0]) < obstacle[1][0]/2 and abs(tmp[1]) < obstacle[1][1]/2:
                return obstacle
        return False

    def hit_test(self, start, end, obstacle):
        """Finds the intersection with a trajectory and obstacles (slow)"""
        for line in self.obstacle_lines(obstacle):
            if self.intersection(line[1], [start, end]):
                return line[0]
        return None

    def render(self, render):
        """Forward the render information"""

        # Update the ball position
        if len(self.ball_frames):
            element = self.ball_frames[0]
            self.ball_frames = self.ball_frames[1:]

            if element is True:
                render.alert((0, 255, 0))
            else:
                self.ball = element

        # Fill the render data and do the render
        render.field_size = self.field_size
        render.goal_position = self.goal_position
        render.goal_radius = self.goal_radius
        render.ball = np.array(self.ball)
        render.obstacles = self.obstacles
        render.score = self.score
        render.iterations = self.iterations
        render.render()


render = GolfRender()
sim = GolfSimulator()
agent = agents.get_agent()(sim, render)
if __name__ == '__main__':
    orientation = 0
    sim.reset_ball()

    print("""--- Mini Golf 

    controls:
     - left/right: move the ball target
     - space: kick the ball
     - r: reset the ball position
     - p: invoke the policy and kick
     - l: start learning
     - s: learning (with steps = 1)
     - q: quit the game
    
    """)

    while True:
        if len(sim.ball_frames) == 0:
            render.arrow = sim.get_kick_target(orientation) / sim.power
        else:
            render.arrow = None
        sim.render(render)

        for key in render.keys():
            if key == pygame.K_LEFT:
                orientation = (orientation-1) % sim.orientations
            if key == pygame.K_RIGHT:
                orientation = (orientation+1) % sim.orientations

            if key == pygame.K_r:
                sim.reset_ball()
            if len(sim.ball_frames) == 0:
                # User action
                if key == pygame.K_SPACE:
                    goal = sim.kick(orientation, True)
                    if goal:
                        sim.reset_ball(True)

                # Agent action
                if key == pygame.K_p:
                    action = agent.pick_action(
                        agent.position_to_state(sim.ball))

                    # Show the decision
                    render.arrow = sim.get_kick_target(action) / sim.power
                    sim.render(render)
                    time.sleep(0.2)

                    # Apply it
                    goal = sim.kick(action, True)
                    if goal:
                        sim.reset_ball(True)

                if key == pygame.K_q:
                    pygame.quit()
            if key == pygame.K_l:
                agent.learn()
            if key == pygame.K_s:
                agent.learn(1)

        if len(sim.ball_frames) == 0:
            # Wrap ball hack
            ball_position = agent.state_to_position(
                agent.position_to_state(sim.ball))
            sim.ball = ball_position

        time.sleep(0.01)
