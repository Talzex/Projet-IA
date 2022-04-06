import time
import pygame.gfxdraw
import pygame
import numpy as np


class GolfRender:
    def __init__(self):
        self.screen = None
        self.size = (800, 400)
        self.field_size = np.array([9, 6])
        self.center_circle_radius = 0.75
        self.goal_position = [0, 0]
        self.goal_radius = 1
        self.ball = np.array([2, 1.5])
        self.arrow = np.array([1, 0])
        self.alert_color = None
        self.alert_start = None
        self.text = []
        self.score = 0
        self.obstacles = []
        self.font = None

    def alert(self, color):
        self.alert_color = color
        self.alert_start = time.time()

    def field_to_screen(self, pos):
        pt = np.array(pos) * self.pixels_per_meter + np.array(self.size)/2.
        return [int(pt[0]), int(pt[1])]

    def render_line(self, pt1, pt2, col=(255, 255, 255), thickness=2):
        pygame.draw.aaline(self.screen, col, self.field_to_screen(
            pt1), self.field_to_screen(pt2), thickness)

    def rot(self, alpha):
        return np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

    def render_ball_and_arrow(self):
        if self.arrow is not None:
            col = (0, 64, 0)
            self.render_line(self.ball, self.ball + self.arrow, col, 5)

            arrowTip = self.ball + self.arrow
            arrowLine = self.arrow
            arrowLine = arrowLine * 0.2 / np.linalg.norm(arrowLine)
            self.render_line(arrowTip, arrowTip +
                             self.rot(2.5).dot(arrowLine), col)
            self.render_line(arrowTip, arrowTip +
                             self.rot(-2.5).dot(arrowLine), col)

        pos = self.field_to_screen(self.ball)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(
            0.08*self.pixels_per_meter), (255, 255, 255))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(
            0.08*self.pixels_per_meter), (255, 255, 255))

    def render_field(self):
        l, h = self.field_size

        col = (130, 200, 20)
        if self.alert_start is not None and time.time() - self.alert_start < 0.5:
            col = self.alert_color
        pygame.draw.rect(self.screen, col,
                         (0, 0, self.size[0], self.size[1]))

        for x in range(0, l+1, 2):
            for y in range(0, h+1):
                pos = self.field_to_screen([x + y % 2 - l/2, y - h/2])
                pygame.draw.rect(self.screen, (130, 220, 20),
                                 (pos[0], pos[1], self.pixels_per_meter, self.pixels_per_meter))

        pos = self.field_to_screen(self.goal_position)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1],
                                     int(self.goal_radius * self.pixels_per_meter), (64, 64, 64))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1],
                                int(self.goal_radius * self.pixels_per_meter), (64, 64, 64))

    def render_obstacles(self):
        for obstacle in self.obstacles:
            x, y = self.field_to_screen(obstacle[0] - obstacle[1]/2)
            w, h = obstacle[1] * self.pixels_per_meter

            pygame.draw.rect(self.screen, (64, 32, 32), (x-2, y-2, w+4, h+4))
        for obstacle in self.obstacles:
            x, y = self.field_to_screen(obstacle[0] - obstacle[1]/2)
            w, h = obstacle[1] * self.pixels_per_meter

            pygame.draw.rect(self.screen, (128, 64, 64), (x, y, w, h))

    def render_text(self):
        if self.font is None:
            self.font = pygame.font.SysFont(None, 14)
        for text in self.text:
            img = self.font.render(text[0], True, text[2])
            pos = self.field_to_screen(text[1])
            self.screen.blit(img, pos)

    def render_score(self):
        font = pygame.font.SysFont(None, 18)
        pos = [0, -self.field_size[1]/2]
        pos = self.field_to_screen(pos)
        img = font.render('Score: %d' % self.score, True, (255, 255, 255))
        self.screen.blit(img, pos)

        if self.iterations:
            img = font.render('Iterations: %d' %
                              self.iterations, True, (255, 255, 255))
            self.screen.blit(img, (pos[0], pos[1]+20))

    def render(self):
        self.pixels_per_meter = (self.size[0] / self.field_size[0])*0.95

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.size, 0, 32)

        self.render_field()
        self.render_text()
        self.render_obstacles()
        self.render_ball_and_arrow()
        self.render_score()

        pygame.display.flip()

    def keys(self):
        keys = []
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                keys.append(event.key)
            elif event.type == pygame.QUIT:
                pygame.quit()
        return keys
