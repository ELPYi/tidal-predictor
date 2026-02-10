"""
Tidal Predictor — Entry point.

Launches the Tkinter-based tidal prediction application.
"""

import tkinter as tk
from gui import TidalPredictorApp


def main():
    root = tk.Tk()
    TidalPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
