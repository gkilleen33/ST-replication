# The Persistent Effect of the African Slave Trades on Development: Replication data and code

This repository contains replication code and data for "The Persistent Effect of the African Slave Trades on Development: Difference-in-difference evidence." 

I make use of a variety of geospatial data sets. Some of these (e.g. LandScan population data) are restricted, so I do not include them. The raster data is also frequently very large. These data sets are used to generate statistics for each polygon in the sample. The data generated from the raw data sets is included, and I present the code used to perform the calculations. All data sources are listed in Appendix A of the included paper.

The file "code/4-analysis.py" contains all analysis, and this can be executed using the included materials in this repository after installing the Python packages listed. It generates each of the tables presented in the paper. Several figures in the Appendix were generated manually in ArcGIS Pro. The code takes several hours to execute because of randomization inference and bootstrapping. The other Python scripts all involve data generation.
