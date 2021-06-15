# IDP-Test
Quick repo for IDP test assignment.

Usage:

The module `border_detection` provides the `detect_borders()` function for Task 1, and the `borders_to_segments()` function for Task 2 (apply it to the output from Task 1).

For testing, adjust comments in the `__main__` section of each module as desired and run that module through python in your preferred shell.

For Task 2, default output is an image/matrix containing the segmented border (and the number of segments), if a *list* of segments is truly necessary as per the task definition, a simple for-loop can compose it from the matrix. Intentionally leaving the output as a raster since it is generally more useful.

Used Python version 3.9.1
