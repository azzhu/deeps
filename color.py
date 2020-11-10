# -----------------colorama模块的一些常量---------------------------
# Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Style: DIM, NORMAL, BRIGHT, RESET_ALL
#

from colorama import init, Fore, Back, Style

init(autoreset=True)


class Colored(object):

    #  前景色:红色  背景色:默认
    @staticmethod
    def red(s):
        return Fore.RED + s + Fore.RESET

    #  前景色:绿色  背景色:默认
    @staticmethod
    def green(s):
        return Fore.GREEN + s + Fore.RESET

    #  前景色:黄色  背景色:默认
    def yellow(self, s):
        return Fore.YELLOW + s + Fore.RESET

    #  前景色:蓝色  背景色:默认
    @staticmethod
    def blue(s):
        return Fore.BLUE + s + Fore.RESET

    #  前景色:洋红色  背景色:默认
    def magenta(self, s):
        return Fore.MAGENTA + s + Fore.RESET

    #  前景色:青色  背景色:默认
    def cyan(self, s):
        return Fore.CYAN + s + Fore.RESET

    #  前景色:白色  背景色:默认
    def white(self, s):
        return Fore.WHITE + s + Fore.RESET

    #  前景色:黑色  背景色:默认
    def black(self, s):
        return Fore.BLACK

    #  前景色:白色  背景色:绿色
    def white_green(self, s):
        return Fore.WHITE + Back.GREEN + s + Fore.RESET + Back.RESET

    #  前景色:黑色  背景色:绿色
    @staticmethod
    def black_green(s):
        return Fore.BLACK + Back.GREEN + s + Fore.RESET + Back.RESET

    #  前景色:黑色  背景色:红色
    @staticmethod
    def black_red(s):
        return Fore.BLACK + Back.RED + s + Fore.RESET + Back.RESET


if __name__ == '__main__':

    color = Colored()
    print(color.red('I am red!'))
    print(color.green('I am gree!'))
    print(color.yellow('I am yellow!'))
    print(color.blue('I am blue!'))
    print(color.magenta('I am magenta!'))
    print(color.cyan('I am cyan!'))
    print(color.white('I am white!'))
    print(color.white_green('I am white green!'))
    print(color.black_green('I am black green!'))
    print(color.black_red('I am black RED!'))