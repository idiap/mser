MSER
===

Linear time Maximally Stable Extremal Regions (MSER) implementation as described
in D. Nistér and H. Stewénius, "Linear Time Maximally Stable Extremal Regions",
ECCV 2008.
The functionality is similar to that of VLFeat MSER feature detector
<http://www.vlfeat.org/overview/mser.html> but the code is several time faster.
MSER is a blob detector, like the Laplacian of Gaussian used by the SIFT
algorithm. It extracts stable connected regions of some level sets from an
image, and optionally fits ellipses to them.
