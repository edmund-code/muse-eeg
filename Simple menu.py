"""
pygame-menu
https://github.com/ppizarror/pygame-menu

EXAMPLE - SIMPLE
Super simple example of pygame-menu usage, featuring a selector and a button.
"""

import pygame_menu
from pygame_menu.examples import create_example_window
import string

from typing import Tuple, Any

surface = create_example_window('Example - Simple', (800, 600))


def set_difficulty(selected: Tuple, value: Any) -> None:
    """
    Set the difficulty of the game.
    """
    print(f'Set difficulty to {selected[0]} ({value})')


def start_the_game() -> None:
    """
    Function that starts a game. This is raised by the menu button,
    here menu can be disabled, etc.
    """
    global user_name
    print(f'{user_name.get_value()}, Do the job here!')


menu = pygame_menu.Menu(
    height=600,
    theme=pygame_menu.themes.THEME_BLUE,
    title='Welcome',
    width=800
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

user_name = menu.add.text_input('Name: ', default='John Doe', maxchar=10)
menu.add.selector('Difficulty: ', alphabet, onchange=set_difficulty)
menu.add.button('Play', start_the_game)
menu.add.button('Quit', pygame_menu.events.EXIT)

if __name__ == '__main__':
    menu.mainloop(surface)