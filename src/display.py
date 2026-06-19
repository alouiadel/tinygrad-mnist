import os
from colorama import init, Fore, Style
from tinygrad import Device

# Initialize colorama for cross-platform terminal color support
init()


# Clears the terminal screen in a cross-platform way (Windows/Unix)
def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


# Positions the cursor at specific coordinates in the terminal
def move_cursor(x, y):
    """Move cursor to position."""
    print(f"\033[{y};{x}H", end="")


# Erases the content of the current terminal line
def clear_line():
    """Clear current line."""
    print("\033[K", end="")


# Displays the training configuration and system information header
def print_header(batch_size, num_steps):
    """Print the header with system information."""
    print(f"{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.BLUE}MNIST Training with TinyGrad".center(60))
    print(f"{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.GREEN}Device:{Fore.RESET} {Device.DEFAULT}")
    print(f"{Fore.GREEN}Batch Size:{Fore.RESET} {batch_size}")
    print(f"{Fore.GREEN}Total Steps:{Fore.RESET} {num_steps}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")


# Renders a visual progress bar with percentage completion
def print_progress_bar(iteration, total, length=50):
    """Print a progress bar."""
    filled = int(length * iteration // total)
    bar = "█" * filled + "░" * (length - filled)
    percent = f"{100 * iteration / total:.1f}%"
    return f"{Fore.CYAN}[{bar}] {percent}{Style.RESET_ALL}"


# Updates and displays the current training metrics and statistics
def print_stats(step, loss, acc, best_acc, elapsed_time, num_steps):
    """Print training statistics."""
    move_cursor(0, 12)
    clear_line()
    print(f"{Fore.GREEN}Step:{Fore.RESET} {step:,}/{num_steps:,}")
    clear_line()
    print(f"{Fore.GREEN}Loss:{Fore.RESET} {loss:.4f}")
    clear_line()
    print(f"{Fore.GREEN}Accuracy:{Fore.RESET} {acc * 100:.2f}%")
    clear_line()
    print(f"{Fore.GREEN}Best Accuracy:{Fore.RESET} {best_acc * 100:.2f}%")
    clear_line()
    print(f"{Fore.GREEN}Time Elapsed:{Fore.RESET} {str(elapsed_time).split('.')[0]}")


# Makes the terminal cursor visible
def show_cursor():
    """Show the cursor."""
    print("\033[?25h", end="")


# Hides the terminal cursor for cleaner display
def hide_cursor():
    """Hide the cursor."""
    print("\033[?25l", end="")
