"""
pygame-menu
pip install pygame-menu -U
"""

import pygame
import pygame_menu
from pygame_menu.examples import create_example_window
import string

from typing import Tuple, Any

from dino_game import *

surface = create_example_window('Brain-Computer-Interface Game using EEG', (600, 200))

def start_the_game() -> None:
    start_game()


menu = pygame_menu.Menu(
    height=200,
    theme=pygame_menu.themes.THEME_BLUE,
    title='Dino Run - EEG',
    width=600
)


chars = list(string.ascii_uppercase)
chars.append(' ')
print(chars)


def createList(r1, r2):
    return list(range(r1, r2+1))

numbers = createList(65,90)
numbers.append(32)
print(numbers)

alphabet = list(zip(chars,numbers))
print(alphabet)

menu.add.button('Play', start_the_game)
menu.add.button('Quit', pygame_menu.events.EXIT)

if __name__ == '__main__':
    menu.mainloop(surface)