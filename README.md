# watchGo
Find a Go board in an image and read the game position. Written with Python, Numpy, and OpenCV.

Cascade classifiers are typically used to identify things like faces or stop signs in an image.
This program uses three different cascade classifiers to find: black stones, white stones, and empty intersections on a Go board.

For more information on training a custom Haar cascade classifier see [here](http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html) and [here](http://note.sonots.com/SciSoftware/haartraining.html).

The function readBoard(image) processes the image and returns a numpy array representing the board, and an array of the coordinates of the four corners of the board.

The function watchBoard() shows a video feed with the corners of the board marked with red dots.
It also shows a diagram of the current board position.
These update after every time motion is detected around the board.
