import pyautogui


def press_scroll_lock():
    """模拟按下 Scroll Lock 键"""
    pyautogui.press('scrolllock')
    # pyautogui.moveTo(100, 100)
    # pyautogui.moveTo(500, 500)
    pyautogui.FAILSAFE = False


def K_to_C(v):
    if isinstance(v, (list, tuple)):
        return [x - 273.15 for x in v]
    return v - 273.15

def C_to_K(v):
    if isinstance(v, (list, tuple)):
        return [x + 273.15 for x in v]
    return v + 273.15