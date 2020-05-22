# ImageMoments
Accurate calculation of raw image moments with O(N+M) multiplications for grayscale images. 

### Raw Image Moments
Moment invariants are a big part of computer vision and pattern recognition. All of them require the calculation of raw image moments, which can be computer directly from the image.
For a grayscale image or pixel array of size (N, M), the calculation of raw image moments typically requires O(NM) mulitplications, this can be very costly for large images (for binary images there are much faster methods, they won't be discussed here). These values are computed using 2d moments of the image. 

### The Discrete Radon Transformation
The DRT reduces this problem from 2d moments of one array, to of 1d moments of 4 projection arrays. The original image can be projected vertically, horizontally and at 45, 135 degrees, and summed along those axis. The raw moments then become linear combinations of 1d moments of these arrays and there is no loss if information. This reduces the number of multiplications from O(NM) to O(N+M)