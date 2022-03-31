# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: ploomber
#     text_representation:
#       extension: .R
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
#   ploomber:
#     injected_manually: true
# ---


# + tags=["parameters"]
upstream = NULL
product = NULL


# + tags=["injected-parameters"]
# Parameters
product = list("nb" = "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\downLoadMetaData.ipynb", "counterparts" = "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\metadata\\counterparts.csv", "countries" = "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\metadata\\countries.csv")

# -

library(rdbnomics)

# get imf metadata
imf_datasets <- rdbnomics::rdb_datasets(provider_code = "IMF")

dataset <- "DOT"
dim <- rdbnomics::rdb_dimensions(provider_code = "IMF", dataset_code = dataset)
countries <- dim$IMF$DOT$REF_AREA
counterpart <- dim$IMF$DOT$COUNTERPART_AREA

write.csv(countries, product$countries)
write.csv(counterpart, product$counterparts)


