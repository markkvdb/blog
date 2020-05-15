---
layout: post
description: I show how you can plot your own map in R using a few lines of code using a pipe-based workflow
categories: [R, Visualisation]
comments: true
---

I show how you can plot your own map in R using a few lines of code using a pipe-based workflow. Several powerful functions of the `sf` packages are presented.

## Analysis

This week I worked on a project for which I needed to create a map plot
with some statistics for selected European countries; I was unfamiliar
with this kind of plots, so I searched online for possible solutions. I
like the **tidyverse** workflow, so I naturally looked for any tutorials
using this style. The first
[hit](http://eriqande.github.io/rep-res-web/lectures/making-maps-with-R.html)
was informative, but it didn't have a high resolution map for Europe.
Furthermore, I like to be able to use any custom map, so I searched for
ways to import a custom map.

[naturalearthdata.com](https://www.naturalearthdata.com) provides many
open-source maps. I decided to select the world map with country borders
on a 1:10m scale (can be found
[here](https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip)).

```R
library(sf)       # For handling geospatial data
library(ggplot2)  # Plotting library
library(dplyr)    # Data manipulation in tidyverse way
library(ggthemes) # Additional themese for the ggplot2 library
library(knitr)    # Nice tables for this document

# This will create a natural-earth subfolder with the map data in the data folder.
if (!file.exists("data/natural-earth")) {
    tmp_file <- tempfile(fileext=".zip")
    download.file("https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_admin_0_countries.zip", 
                tmp_file)
    unzip(tmp_file, exdir = "data/natural-earth")
}
```

Importing these maps, however, was not straightforward to me. [These
lecture
slides](https://cran.r-project.org/doc/contrib/intro-spatial-rl.pdf)
provides a way to import custom maps, but the syntax of the `sp` package
seems very untuitive with `S4` objects for the polygons. Furthermore,
the `SpatialDataFrame` objects do not support a pipe-based workflow.
However, [this tutorial](https://edzer.github.io/UseR2017/) presents how
the modern `sf` package can be used to manipulate, plot and import
spatial data in a tidyverse manner.

Importing our world map is as easy as
```R
map_data <- st_read("data/natural-earth/", "ne_10m_admin_0_countries")
```

```
## Reading layer `ne_10m_admin_0_countries' from data source `Map-Plotting/data/natural-earth' using driver `ESRI Shapefile'
## Simple feature collection with 255 features and 94 fields
## geometry type:  MULTIPOLYGON
## dimension:      XY
## bbox:           xmin: -180 ymin: -90 xmax: 180 ymax: 83.6341
## epsg (SRID):    4326
## proj4string:    +proj=longlat +datum=WGS84 +no_defs
```

The `map_data` uses `data.frame`s for its features and saves the
geometric features as a list in the column `geometry`. We can now easily
explore the data in `map_data`, e.g.,

```R
features_map_data <- map_data %>%
    as_tibble() %>%
    select(-geometry) %>%
    head(10)

kable(features_map_data)
```

<table>
<thead>
<tr class="header">
<th align="left">featurecla</th>
<th align="right">scalerank</th>
<th align="right">LABELRANK</th>
<th align="left">SOVEREIGNT</th>
<th align="left">SOV_A3</th>
<th align="right">ADM0_DIF</th>
<th align="right">LEVEL</th>
<th align="left">TYPE</th>
<th align="left">ADMIN</th>
<th align="left">ADM0_A3</th>
<th align="right">GEOU_DIF</th>
<th align="left">GEOUNIT</th>
<th align="left">GU_A3</th>
<th align="right">SU_DIF</th>
<th align="left">SUBUNIT</th>
<th align="left">SU_A3</th>
<th align="right">BRK_DIFF</th>
<th align="left">NAME</th>
<th align="left">NAME_LONG</th>
<th align="left">BRK_A3</th>
<th align="left">BRK_NAME</th>
<th align="left">BRK_GROUP</th>
<th align="left">ABBREV</th>
<th align="left">POSTAL</th>
<th align="left">FORMAL_EN</th>
<th align="left">FORMAL_FR</th>
<th align="left">NAME_CIAWF</th>
<th align="left">NOTE_ADM0</th>
<th align="left">NOTE_BRK</th>
<th align="left">NAME_SORT</th>
<th align="left">NAME_ALT</th>
<th align="right">MAPCOLOR7</th>
<th align="right">MAPCOLOR8</th>
<th align="right">MAPCOLOR9</th>
<th align="right">MAPCOLOR13</th>
<th align="right">POP_EST</th>
<th align="right">POP_RANK</th>
<th align="right">GDP_MD_EST</th>
<th align="right">POP_YEAR</th>
<th align="right">LASTCENSUS</th>
<th align="right">GDP_YEAR</th>
<th align="left">ECONOMY</th>
<th align="left">INCOME_GRP</th>
<th align="right">WIKIPEDIA</th>
<th align="left">FIPS_10_</th>
<th align="left">ISO_A2</th>
<th align="left">ISO_A3</th>
<th align="left">ISO_A3_EH</th>
<th align="left">ISO_N3</th>
<th align="left">UN_A3</th>
<th align="left">WB_A2</th>
<th align="left">WB_A3</th>
<th align="right">WOE_ID</th>
<th align="right">WOE_ID_EH</th>
<th align="left">WOE_NOTE</th>
<th align="left">ADM0_A3_IS</th>
<th align="left">ADM0_A3_US</th>
<th align="right">ADM0_A3_UN</th>
<th align="right">ADM0_A3_WB</th>
<th align="left">CONTINENT</th>
<th align="left">REGION_UN</th>
<th align="left">SUBREGION</th>
<th align="left">REGION_WB</th>
<th align="right">NAME_LEN</th>
<th align="right">LONG_LEN</th>
<th align="right">ABBREV_LEN</th>
<th align="right">TINY</th>
<th align="right">HOMEPART</th>
<th align="right">MIN_ZOOM</th>
<th align="right">MIN_LABEL</th>
<th align="right">MAX_LABEL</th>
<th align="right">NE_ID</th>
<th align="left">WIKIDATAID</th>
<th align="left">NAME_AR</th>
<th align="left">NAME_BN</th>
<th align="left">NAME_DE</th>
<th align="left">NAME_EN</th>
<th align="left">NAME_ES</th>
<th align="left">NAME_FR</th>
<th align="left">NAME_EL</th>
<th align="left">NAME_HI</th>
<th align="left">NAME_HU</th>
<th align="left">NAME_ID</th>
<th align="left">NAME_IT</th>
<th align="left">NAME_JA</th>
<th align="left">NAME_KO</th>
<th align="left">NAME_NL</th>
<th align="left">NAME_PL</th>
<th align="left">NAME_PT</th>
<th align="left">NAME_RU</th>
<th align="left">NAME_SV</th>
<th align="left">NAME_TR</th>
<th align="left">NAME_VI</th>
<th align="left">NAME_ZH</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left">Admin-0 country</td>
<td align="right">5</td>
<td align="right">2</td>
<td align="left">Indonesia</td>
<td align="left">IDN</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">Sovereign country</td>
<td align="left">Indonesia</td>
<td align="left">IDN</td>
<td align="right">0</td>
<td align="left">Indonesia</td>
<td align="left">IDN</td>
<td align="right">0</td>
<td align="left">Indonesia</td>
<td align="left">IDN</td>
<td align="right">0</td>
<td align="left">Indonesia</td>
<td align="left">Indonesia</td>
<td align="left">IDN</td>
<td align="left">Indonesia</td>
<td align="left">NA</td>
<td align="left">Indo.</td>
<td align="left">INDO</td>
<td align="left">Republic of Indonesia</td>
<td align="left">NA</td>
<td align="left">Indonesia</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">Indonesia</td>
<td align="left">NA</td>
<td align="right">6</td>
<td align="right">6</td>
<td align="right">6</td>
<td align="right">11</td>
<td align="right">260580739</td>
<td align="right">17</td>
<td align="right">3028000</td>
<td align="right">2017</td>
<td align="right">2010</td>
<td align="right">2016</td>
<td align="left">4. Emerging region: MIKT</td>
<td align="left">4. Lower middle income</td>
<td align="right">-99</td>
<td align="left">ID</td>
<td align="left">ID</td>
<td align="left">IDN</td>
<td align="left">IDN</td>
<td align="left">360</td>
<td align="left">360</td>
<td align="left">ID</td>
<td align="left">IDN</td>
<td align="right">23424846</td>
<td align="right">23424846</td>
<td align="left">Exact WOE match as country</td>
<td align="left">IDN</td>
<td align="left">IDN</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">Asia</td>
<td align="left">Asia</td>
<td align="left">South-Eastern Asia</td>
<td align="left">East Asia &amp; Pacific</td>
<td align="right">9</td>
<td align="right">9</td>
<td align="right">5</td>
<td align="right">-99</td>
<td align="right">1</td>
<td align="right">0</td>
<td align="right">1.7</td>
<td align="right">6.7</td>
<td align="right">1159320845</td>
<td align="left">Q252</td>
<td align="left">إندونيسيا</td>
<td align="left">ইন্দোনেশিয়া</td>
<td align="left">Indonesien</td>
<td align="left">Indonesia</td>
<td align="left">Indonesia</td>
<td align="left">Indonésie</td>
<td align="left">Ινδονησία</td>
<td align="left">इंडोनेशिया</td>
<td align="left">Indonézi</td>
<td align="left">a Indonesia</td>
<td align="left">Indonesia</td>
<td align="left">インドネシア</td>
<td align="left">인도네시아</td>
<td align="left">Indonesië</td>
<td align="left">Indonezja</td>
<td align="left">Indonési</td>
<td align="left">a Индонезия</td>
<td align="left">Indonesie</td>
<td align="left">n Endonezya</td>
<td align="left">Indonesia</td>
<td align="left">印度尼西亚</td>
</tr>
<tr class="even">
<td align="left">Admin-0 country</td>
<td align="right">5</td>
<td align="right">3</td>
<td align="left">Malaysia</td>
<td align="left">MYS</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">Sovereign country</td>
<td align="left">Malaysia</td>
<td align="left">MYS</td>
<td align="right">0</td>
<td align="left">Malaysia</td>
<td align="left">MYS</td>
<td align="right">0</td>
<td align="left">Malaysia</td>
<td align="left">MYS</td>
<td align="right">0</td>
<td align="left">Malaysia</td>
<td align="left">Malaysia</td>
<td align="left">MYS</td>
<td align="left">Malaysia</td>
<td align="left">NA</td>
<td align="left">Malay.</td>
<td align="left">MY</td>
<td align="left">Malaysia</td>
<td align="left">NA</td>
<td align="left">Malaysia</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">Malaysia</td>
<td align="left">NA</td>
<td align="right">2</td>
<td align="right">4</td>
<td align="right">3</td>
<td align="right">6</td>
<td align="right">31381992</td>
<td align="right">15</td>
<td align="right">863000</td>
<td align="right">2017</td>
<td align="right">2010</td>
<td align="right">2016</td>
<td align="left">6. Developing region</td>
<td align="left">3. Upper middle income</td>
<td align="right">-99</td>
<td align="left">MY</td>
<td align="left">MY</td>
<td align="left">MYS</td>
<td align="left">MYS</td>
<td align="left">458</td>
<td align="left">458</td>
<td align="left">MY</td>
<td align="left">MYS</td>
<td align="right">23424901</td>
<td align="right">23424901</td>
<td align="left">Exact WOE match as country</td>
<td align="left">MYS</td>
<td align="left">MYS</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">Asia</td>
<td align="left">Asia</td>
<td align="left">South-Eastern Asia</td>
<td align="left">East Asia &amp; Pacific</td>
<td align="right">8</td>
<td align="right">8</td>
<td align="right">6</td>
<td align="right">-99</td>
<td align="right">1</td>
<td align="right">0</td>
<td align="right">3.0</td>
<td align="right">8.0</td>
<td align="right">1159321083</td>
<td align="left">Q833</td>
<td align="left">ماليزيا</td>
<td align="left">মালয়েশিয়া</td>
<td align="left">Malaysia</td>
<td align="left">Malaysia</td>
<td align="left">Malasia</td>
<td align="left">Malaisie</td>
<td align="left">Μαλαισία</td>
<td align="left">मलेशिया</td>
<td align="left">Malajzia</td>
<td align="left">Malaysia</td>
<td align="left">Malesia</td>
<td align="left">マレーシア</td>
<td align="left">말레이시아</td>
<td align="left">Maleisië</td>
<td align="left">Malezja</td>
<td align="left">Malásia</td>
<td align="left">Малайзия</td>
<td align="left">Malaysia</td>
<td align="left">Malezya</td>
<td align="left">Malaysia</td>
<td align="left">马来西亚</td>
</tr>
<tr class="odd">
<td align="left">Admin-0 country</td>
<td align="right">6</td>
<td align="right">2</td>
<td align="left">Chile</td>
<td align="left">CHL</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">Sovereign country</td>
<td align="left">Chile</td>
<td align="left">CHL</td>
<td align="right">0</td>
<td align="left">Chile</td>
<td align="left">CHL</td>
<td align="right">0</td>
<td align="left">Chile</td>
<td align="left">CHL</td>
<td align="right">0</td>
<td align="left">Chile</td>
<td align="left">Chile</td>
<td align="left">CHL</td>
<td align="left">Chile</td>
<td align="left">NA</td>
<td align="left">Chile</td>
<td align="left">CL</td>
<td align="left">Republic of Chile</td>
<td align="left">NA</td>
<td align="left">Chile</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">Chile</td>
<td align="left">NA</td>
<td align="right">5</td>
<td align="right">1</td>
<td align="right">5</td>
<td align="right">9</td>
<td align="right">17789267</td>
<td align="right">14</td>
<td align="right">436100</td>
<td align="right">2017</td>
<td align="right">2002</td>
<td align="right">2016</td>
<td align="left">5. Emerging region: G20</td>
<td align="left">3. Upper middle income</td>
<td align="right">-99</td>
<td align="left">CI</td>
<td align="left">CL</td>
<td align="left">CHL</td>
<td align="left">CHL</td>
<td align="left">152</td>
<td align="left">152</td>
<td align="left">CL</td>
<td align="left">CHL</td>
<td align="right">23424782</td>
<td align="right">23424782</td>
<td align="left">Exact WOE match as country</td>
<td align="left">CHL</td>
<td align="left">CHL</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">South America</td>
<td align="left">Americas</td>
<td align="left">South America</td>
<td align="left">Latin America &amp; Caribbean</td>
<td align="right">5</td>
<td align="right">5</td>
<td align="right">5</td>
<td align="right">-99</td>
<td align="right">1</td>
<td align="right">0</td>
<td align="right">1.7</td>
<td align="right">6.7</td>
<td align="right">1159320493</td>
<td align="left">Q298</td>
<td align="left">تشيلي</td>
<td align="left">চিলি</td>
<td align="left">Chile</td>
<td align="left">Chile</td>
<td align="left">Chile</td>
<td align="left">Chili</td>
<td align="left">Χιλή</td>
<td align="left">चिली</td>
<td align="left">Chile</td>
<td align="left">Chili</td>
<td align="left">Cile</td>
<td align="left">チリ</td>
<td align="left">칠레</td>
<td align="left">Chili</td>
<td align="left">Chile</td>
<td align="left">Chile</td>
<td align="left">Чили</td>
<td align="left">Chile</td>
<td align="left">Şili</td>
<td align="left">Chile</td>
<td align="left">智利</td>
</tr>
<tr class="even">
<td align="left">Admin-0 country</td>
<td align="right">0</td>
<td align="right">3</td>
<td align="left">Bolivia</td>
<td align="left">BOL</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">Sovereign country</td>
<td align="left">Bolivia</td>
<td align="left">BOL</td>
<td align="right">0</td>
<td align="left">Bolivia</td>
<td align="left">BOL</td>
<td align="right">0</td>
<td align="left">Bolivia</td>
<td align="left">BOL</td>
<td align="right">0</td>
<td align="left">Bolivia</td>
<td align="left">Bolivia</td>
<td align="left">BOL</td>
<td align="left">Bolivia</td>
<td align="left">NA</td>
<td align="left">Bolivia</td>
<td align="left">BO</td>
<td align="left">Plurinational State of Bolivia</td>
<td align="left">NA</td>
<td align="left">Bolivia</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">Bolivia</td>
<td align="left">NA</td>
<td align="right">1</td>
<td align="right">5</td>
<td align="right">2</td>
<td align="right">3</td>
<td align="right">11138234</td>
<td align="right">14</td>
<td align="right">78350</td>
<td align="right">2017</td>
<td align="right">2001</td>
<td align="right">2016</td>
<td align="left">5. Emerging region: G20</td>
<td align="left">4. Lower middle income</td>
<td align="right">-99</td>
<td align="left">BL</td>
<td align="left">BO</td>
<td align="left">BOL</td>
<td align="left">BOL</td>
<td align="left">068</td>
<td align="left">068</td>
<td align="left">BO</td>
<td align="left">BOL</td>
<td align="right">23424762</td>
<td align="right">23424762</td>
<td align="left">Exact WOE match as country</td>
<td align="left">BOL</td>
<td align="left">BOL</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">South America</td>
<td align="left">Americas</td>
<td align="left">South America</td>
<td align="left">Latin America &amp; Caribbean</td>
<td align="right">7</td>
<td align="right">7</td>
<td align="right">7</td>
<td align="right">-99</td>
<td align="right">1</td>
<td align="right">0</td>
<td align="right">3.0</td>
<td align="right">7.5</td>
<td align="right">1159320439</td>
<td align="left">Q750</td>
<td align="left">بوليفيا</td>
<td align="left">বলিভিয়া</td>
<td align="left">Bolivien</td>
<td align="left">Bolivia</td>
<td align="left">Bolivia</td>
<td align="left">Bolivie</td>
<td align="left">Βολιβία</td>
<td align="left">बोलिविया</td>
<td align="left">Bolívia</td>
<td align="left">Bolivia</td>
<td align="left">Bolivia</td>
<td align="left">ボリビア</td>
<td align="left">볼리비아</td>
<td align="left">Bolivia</td>
<td align="left">Boliwia</td>
<td align="left">Bolívia</td>
<td align="left">Боливия</td>
<td align="left">Bolivia</td>
<td align="left">Bolivya</td>
<td align="left">Bolivia</td>
<td align="left">玻利維亞</td>
</tr>
<tr class="odd">
<td align="left">Admin-0 country</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">Peru</td>
<td align="left">PER</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">Sovereign country</td>
<td align="left">Peru</td>
<td align="left">PER</td>
<td align="right">0</td>
<td align="left">Peru</td>
<td align="left">PER</td>
<td align="right">0</td>
<td align="left">Peru</td>
<td align="left">PER</td>
<td align="right">0</td>
<td align="left">Peru</td>
<td align="left">Peru</td>
<td align="left">PER</td>
<td align="left">Peru</td>
<td align="left">NA</td>
<td align="left">Peru</td>
<td align="left">PE</td>
<td align="left">Republic of Peru</td>
<td align="left">NA</td>
<td align="left">Peru</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">Peru</td>
<td align="left">NA</td>
<td align="right">4</td>
<td align="right">4</td>
<td align="right">4</td>
<td align="right">11</td>
<td align="right">31036656</td>
<td align="right">15</td>
<td align="right">410400</td>
<td align="right">2017</td>
<td align="right">2007</td>
<td align="right">2016</td>
<td align="left">5. Emerging region: G20</td>
<td align="left">3. Upper middle income</td>
<td align="right">-99</td>
<td align="left">PE</td>
<td align="left">PE</td>
<td align="left">PER</td>
<td align="left">PER</td>
<td align="left">604</td>
<td align="left">604</td>
<td align="left">PE</td>
<td align="left">PER</td>
<td align="right">23424919</td>
<td align="right">23424919</td>
<td align="left">Exact WOE match as country</td>
<td align="left">PER</td>
<td align="left">PER</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">South America</td>
<td align="left">Americas</td>
<td align="left">South America</td>
<td align="left">Latin America &amp; Caribbean</td>
<td align="right">4</td>
<td align="right">4</td>
<td align="right">4</td>
<td align="right">-99</td>
<td align="right">1</td>
<td align="right">0</td>
<td align="right">2.0</td>
<td align="right">7.0</td>
<td align="right">1159321163</td>
<td align="left">Q419</td>
<td align="left">بيرو</td>
<td align="left">পেরু</td>
<td align="left">Peru</td>
<td align="left">Peru</td>
<td align="left">Perú</td>
<td align="left">Pérou</td>
<td align="left">Περού</td>
<td align="left">पेरू</td>
<td align="left">Peru</td>
<td align="left">Peru</td>
<td align="left">Perù</td>
<td align="left">ペルー</td>
<td align="left">페루</td>
<td align="left">Peru</td>
<td align="left">Peru</td>
<td align="left">Peru</td>
<td align="left">Перу</td>
<td align="left">Peru</td>
<td align="left">Peru</td>
<td align="left">Peru</td>
<td align="left">秘鲁</td>
</tr>
<tr class="even">
<td align="left">Admin-0 country</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">Argentina</td>
<td align="left">ARG</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">Sovereign country</td>
<td align="left">Argentina</td>
<td align="left">ARG</td>
<td align="right">0</td>
<td align="left">Argentina</td>
<td align="left">ARG</td>
<td align="right">0</td>
<td align="left">Argentina</td>
<td align="left">ARG</td>
<td align="right">0</td>
<td align="left">Argentina</td>
<td align="left">Argentina</td>
<td align="left">ARG</td>
<td align="left">Argentina</td>
<td align="left">NA</td>
<td align="left">Arg.</td>
<td align="left">AR</td>
<td align="left">Argentine Republic</td>
<td align="left">NA</td>
<td align="left">Argentina</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">Argentina</td>
<td align="left">NA</td>
<td align="right">3</td>
<td align="right">1</td>
<td align="right">3</td>
<td align="right">13</td>
<td align="right">44293293</td>
<td align="right">15</td>
<td align="right">879400</td>
<td align="right">2017</td>
<td align="right">2010</td>
<td align="right">2016</td>
<td align="left">5. Emerging region: G20</td>
<td align="left">3. Upper middle income</td>
<td align="right">-99</td>
<td align="left">AR</td>
<td align="left">AR</td>
<td align="left">ARG</td>
<td align="left">ARG</td>
<td align="left">032</td>
<td align="left">032</td>
<td align="left">AR</td>
<td align="left">ARG</td>
<td align="right">23424747</td>
<td align="right">23424747</td>
<td align="left">Exact WOE match as country</td>
<td align="left">ARG</td>
<td align="left">ARG</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">South America</td>
<td align="left">Americas</td>
<td align="left">South America</td>
<td align="left">Latin America &amp; Caribbean</td>
<td align="right">9</td>
<td align="right">9</td>
<td align="right">4</td>
<td align="right">-99</td>
<td align="right">1</td>
<td align="right">0</td>
<td align="right">2.0</td>
<td align="right">7.0</td>
<td align="right">1159320331</td>
<td align="left">Q414</td>
<td align="left">الأرجنتين</td>
<td align="left">আর্জেন্টিনা</td>
<td align="left">Argentinien</td>
<td align="left">Argentina</td>
<td align="left">Argentina</td>
<td align="left">Argentine</td>
<td align="left">Αργεντινή</td>
<td align="left">अर्जेण्टीना</td>
<td align="left">Argentí</td>
<td align="left">na Argentina</td>
<td align="left">Argentina</td>
<td align="left">アルゼンチン</td>
<td align="left">아르헨티나</td>
<td align="left">Argentinië</td>
<td align="left">Argentyna</td>
<td align="left">Argenti</td>
<td align="left">na Аргентина</td>
<td align="left">Argentin</td>
<td align="left">a Arjantin</td>
<td align="left">Argentina</td>
<td align="left">阿根廷</td>
</tr>
<tr class="odd">
<td align="left">Admin-0 country</td>
<td align="right">3</td>
<td align="right">3</td>
<td align="left">United Kingdom</td>
<td align="left">GB1</td>
<td align="right">1</td>
<td align="right">2</td>
<td align="left">Dependency</td>
<td align="left">Dhekelia Sovereign Base Area</td>
<td align="left">ESB</td>
<td align="right">0</td>
<td align="left">Dhekelia Sovereign Base Area</td>
<td align="left">ESB</td>
<td align="right">0</td>
<td align="left">Dhekelia Sovereign Base Area</td>
<td align="left">ESB</td>
<td align="right">0</td>
<td align="left">Dhekelia</td>
<td align="left">Dhekelia</td>
<td align="left">ESB</td>
<td align="left">Dhekelia</td>
<td align="left">NA</td>
<td align="left">Dhek.</td>
<td align="left">DH</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">U.K. Base</td>
<td align="left">NA</td>
<td align="left">Dhekelia Sovereign Base Area</td>
<td align="left">NA</td>
<td align="right">6</td>
<td align="right">6</td>
<td align="right">6</td>
<td align="right">3</td>
<td align="right">7850</td>
<td align="right">5</td>
<td align="right">314</td>
<td align="right">2013</td>
<td align="right">-99</td>
<td align="right">2013</td>
<td align="left">2. Developed region: nonG7</td>
<td align="left">2. High income: nonOECD</td>
<td align="right">-99</td>
<td align="left">-99</td>
<td align="left">-99</td>
<td align="left">-99</td>
<td align="left">-99</td>
<td align="left">-99</td>
<td align="left">-099</td>
<td align="left">-99</td>
<td align="left">-99</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">No WOE equivalent.</td>
<td align="left">GBR</td>
<td align="left">ESB</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">Asia</td>
<td align="left">Asia</td>
<td align="left">Western Asia</td>
<td align="left">Europe &amp; Central Asia</td>
<td align="right">8</td>
<td align="right">8</td>
<td align="right">5</td>
<td align="right">3</td>
<td align="right">-99</td>
<td align="right">0</td>
<td align="right">6.5</td>
<td align="right">11.0</td>
<td align="right">1159320709</td>
<td align="left">Q9206745</td>
<td align="left">ديكيليا كانتونمنت</td>
<td align="left">দেখেলিয়া ক্যান্টনমেন্</td>
<td align="left">ট Dekelia</td>
<td align="left">Dhekelia Cantonment</td>
<td align="left">Dekelia</td>
<td align="left">Dhekelia</td>
<td align="left">Ντεκέλια Κάντονμεντ</td>
<td align="left">ढेकेलिया छावनी</td>
<td align="left">Dekéli</td>
<td align="left">a Dhekelia Cantonment</td>
<td align="left">Base di Dheke</td>
<td align="left">lia デケリア</td>
<td align="left">데켈리아 지</td>
<td align="left">역 Dhekelia Cantonme</td>
<td align="left">nt Dhekelia</td>
<td align="left">Dekeli</td>
<td align="left">a Декелия</td>
<td align="left">Dhekeli</td>
<td align="left">a Dhekelia Kantonu</td>
<td align="left">Căn cứ quân sự Dhekelia</td>
<td align="left">NA</td>
</tr>
<tr class="even">
<td align="left">Admin-0 country</td>
<td align="right">6</td>
<td align="right">5</td>
<td align="left">Cyprus</td>
<td align="left">CYP</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">Sovereign country</td>
<td align="left">Cyprus</td>
<td align="left">CYP</td>
<td align="right">0</td>
<td align="left">Cyprus</td>
<td align="left">CYP</td>
<td align="right">0</td>
<td align="left">Cyprus</td>
<td align="left">CYP</td>
<td align="right">0</td>
<td align="left">Cyprus</td>
<td align="left">Cyprus</td>
<td align="left">CYP</td>
<td align="left">Cyprus</td>
<td align="left">NA</td>
<td align="left">Cyp.</td>
<td align="left">CY</td>
<td align="left">Republic of Cyprus</td>
<td align="left">NA</td>
<td align="left">Cyprus</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">Cyprus</td>
<td align="left">NA</td>
<td align="right">1</td>
<td align="right">2</td>
<td align="right">3</td>
<td align="right">7</td>
<td align="right">1221549</td>
<td align="right">12</td>
<td align="right">29260</td>
<td align="right">2017</td>
<td align="right">2001</td>
<td align="right">2016</td>
<td align="left">6. Developing region</td>
<td align="left">2. High income: nonOECD</td>
<td align="right">-99</td>
<td align="left">CY</td>
<td align="left">CY</td>
<td align="left">CYP</td>
<td align="left">CYP</td>
<td align="left">196</td>
<td align="left">196</td>
<td align="left">CY</td>
<td align="left">CYP</td>
<td align="right">-90</td>
<td align="right">23424994</td>
<td align="left">WOE lists as subunit of united Cyprus</td>
<td align="left">CYP</td>
<td align="left">CYP</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">Asia</td>
<td align="left">Asia</td>
<td align="left">Western Asia</td>
<td align="left">Europe &amp; Central Asia</td>
<td align="right">6</td>
<td align="right">6</td>
<td align="right">4</td>
<td align="right">-99</td>
<td align="right">1</td>
<td align="right">0</td>
<td align="right">4.5</td>
<td align="right">9.5</td>
<td align="right">1159320533</td>
<td align="left">Q229</td>
<td align="left">قبرص</td>
<td align="left">সাইপ্রাস</td>
<td align="left">Republik Zypern</td>
<td align="left">Cyprus</td>
<td align="left">Chipre</td>
<td align="left">Chypre</td>
<td align="left">Κύπρος</td>
<td align="left">साइप्रस</td>
<td align="left">Ciprus</td>
<td align="left">Siprus</td>
<td align="left">Cipro</td>
<td align="left">キプロス</td>
<td align="left">키프로스</td>
<td align="left">Cyprus</td>
<td align="left">Cypr</td>
<td align="left">Chipre</td>
<td align="left">Кипр</td>
<td align="left">Cypern</td>
<td align="left">Kıbrıs Cumhuriyeti</td>
<td align="left">Cộng hòa Síp</td>
<td align="left">賽普勒斯</td>
</tr>
<tr class="odd">
<td align="left">Admin-0 country</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">India</td>
<td align="left">IND</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">Sovereign country</td>
<td align="left">India</td>
<td align="left">IND</td>
<td align="right">0</td>
<td align="left">India</td>
<td align="left">IND</td>
<td align="right">0</td>
<td align="left">India</td>
<td align="left">IND</td>
<td align="right">0</td>
<td align="left">India</td>
<td align="left">India</td>
<td align="left">IND</td>
<td align="left">India</td>
<td align="left">NA</td>
<td align="left">India</td>
<td align="left">IND</td>
<td align="left">Republic of India</td>
<td align="left">NA</td>
<td align="left">India</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">India</td>
<td align="left">NA</td>
<td align="right">1</td>
<td align="right">3</td>
<td align="right">2</td>
<td align="right">2</td>
<td align="right">1281935911</td>
<td align="right">18</td>
<td align="right">8721000</td>
<td align="right">2017</td>
<td align="right">2011</td>
<td align="right">2016</td>
<td align="left">3. Emerging region: BRIC</td>
<td align="left">4. Lower middle income</td>
<td align="right">-99</td>
<td align="left">IN</td>
<td align="left">IN</td>
<td align="left">IND</td>
<td align="left">IND</td>
<td align="left">356</td>
<td align="left">356</td>
<td align="left">IN</td>
<td align="left">IND</td>
<td align="right">23424848</td>
<td align="right">23424848</td>
<td align="left">Exact WOE match as country</td>
<td align="left">IND</td>
<td align="left">IND</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">Asia</td>
<td align="left">Asia</td>
<td align="left">Southern Asia</td>
<td align="left">South Asia</td>
<td align="right">5</td>
<td align="right">5</td>
<td align="right">5</td>
<td align="right">-99</td>
<td align="right">1</td>
<td align="right">0</td>
<td align="right">1.7</td>
<td align="right">6.7</td>
<td align="right">1159320847</td>
<td align="left">Q668</td>
<td align="left">الهند</td>
<td align="left">ভারত</td>
<td align="left">Indien</td>
<td align="left">India</td>
<td align="left">India</td>
<td align="left">Inde</td>
<td align="left">Ινδία</td>
<td align="left">भारत</td>
<td align="left">India</td>
<td align="left">India</td>
<td align="left">India</td>
<td align="left">インド</td>
<td align="left">인도</td>
<td align="left">India</td>
<td align="left">Indie</td>
<td align="left">Índia</td>
<td align="left">Индия</td>
<td align="left">Indien</td>
<td align="left">Hindistan</td>
<td align="left">Ấn Độ</td>
<td align="left">印度</td>
</tr>
<tr class="even">
<td align="left">Admin-0 country</td>
<td align="right">0</td>
<td align="right">2</td>
<td align="left">China</td>
<td align="left">CH1</td>
<td align="right">1</td>
<td align="right">2</td>
<td align="left">Country</td>
<td align="left">China</td>
<td align="left">CHN</td>
<td align="right">0</td>
<td align="left">China</td>
<td align="left">CHN</td>
<td align="right">0</td>
<td align="left">China</td>
<td align="left">CHN</td>
<td align="right">0</td>
<td align="left">China</td>
<td align="left">China</td>
<td align="left">CHN</td>
<td align="left">China</td>
<td align="left">NA</td>
<td align="left">China</td>
<td align="left">CN</td>
<td align="left">People's Republic of China</td>
<td align="left">NA</td>
<td align="left">China</td>
<td align="left">NA</td>
<td align="left">NA</td>
<td align="left">China</td>
<td align="left">NA</td>
<td align="right">4</td>
<td align="right">4</td>
<td align="right">4</td>
<td align="right">3</td>
<td align="right">1379302771</td>
<td align="right">18</td>
<td align="right">21140000</td>
<td align="right">2017</td>
<td align="right">2010</td>
<td align="right">2016</td>
<td align="left">3. Emerging region: BRIC</td>
<td align="left">3. Upper middle income</td>
<td align="right">-99</td>
<td align="left">CH</td>
<td align="left">CN</td>
<td align="left">CHN</td>
<td align="left">CHN</td>
<td align="left">156</td>
<td align="left">156</td>
<td align="left">CN</td>
<td align="left">CHN</td>
<td align="right">23424781</td>
<td align="right">23424781</td>
<td align="left">Exact WOE match as country</td>
<td align="left">CHN</td>
<td align="left">CHN</td>
<td align="right">-99</td>
<td align="right">-99</td>
<td align="left">Asia</td>
<td align="left">Asia</td>
<td align="left">Eastern Asia</td>
<td align="left">East Asia &amp; Pacific</td>
<td align="right">5</td>
<td align="right">5</td>
<td align="right">5</td>
<td align="right">-99</td>
<td align="right">1</td>
<td align="right">0</td>
<td align="right">1.7</td>
<td align="right">5.7</td>
<td align="right">1159320471</td>
<td align="left">Q148</td>
<td align="left">جمهورية الصين الشعبية</td>
<td align="left">গণপ্রজাতন্ত্রী চীন</td>
<td align="left">Volksrepublik China</td>
<td align="left">People's Republic of China</td>
<td align="left">República Popular China</td>
<td align="left">République populaire de Chine</td>
<td align="left">Λαϊκή Δημοκρατία της Κίνας</td>
<td align="left">चीनी जनवादी गणराज्</td>
<td align="left">य Kína</td>
<td align="left">Republik Rakyat Tiongko</td>
<td align="left">k Cina</td>
<td align="left">中華人民共和国</td>
<td align="left">중화인민공화국</td>
<td align="left">Volksrepubliek China</td>
<td align="left">Chińska Republika Ludowa</td>
<td align="left">China</td>
<td align="left">Китайская Народная Республика</td>
<td align="left">Kina</td>
<td align="left">Çin Halk Cumhuriyeti</td>
<td align="left">Cộng hòa Nhân dân Trung Hoa</td>
<td align="left">中华人民共和国</td>
</tr>
</tbody>
</table>

For this tutorial we want to focus on a European countries, hence we
need to filter the data to only contain the european countries' info.
Fortunately, the `map_data` contains a feature `CONTINTENT`, so we can
easily filter out the unwanted countries.

```R
europe_map_data <- map_data %>%
    select(NAME, CONTINENT, SUBREGION, POP_EST) %>%
    filter(CONTINENT == "Europe")
```

Lets try to plot a map of European countries. New versions of `ggplot2`
contain a function `geom_sf` which supports plotting `sf` objects
directly, so lets try it...

```R
ggplot(europe_map_data) + geom_sf() +
    theme_minimal()
```

![]({{ site.baseurl }}/images/map_article_2018_files/figure-markdown_strict/first_attempt_plot-1.png)

That does not seem to work... the reason is that, even though we removed
the data of non European countries, we never changed the `bbox` setting
of our data. The `bbox` object sets the longitude and latitude range for
our plot, which is still for the whole europe. To change this we can use
the `st_crop` function as

```R
europe_map_data <- europe_map_data %>%
    st_crop(xmin=-25, xmax=55, ymin=35, ymax=71)

## although coordinates are longitude/latitude, st_intersection assumes that they are planar

## Warning: attribute variables are assumed to be spatially constant
## throughout all geometries

ggplot(europe_map_data) + geom_sf() +
    theme_minimal()
```

![]({{ site.baseurl }}/images/map_article_2018_files/figure-markdown_strict/crop_box-1.png)

If you're familiar with the `ggplot2` workflow, it is now easy to
construct the aesthetic mappings like you're used to. Our `map_data`
contains a feature `SUBREGION` and Europe is divided into Northern,
Eastern, Southern and Western Europe. We can easily visualise this in
our European map as

```R
ggplot(europe_map_data) + geom_sf(aes(fill=SUBREGION)) +
    theme_minimal()
```

![]({{ site.baseurl }}/images/map_article_2018_files/figure-markdown_strict/europe_map_divide-1.png)

The `sf` has many in-built functions; one of these functions is
`st_area` which can be used to compute the area of polygons. The
population density of each country can be easily plotted by

```R
europe_map_data <- europe_map_data %>%
    mutate(area = as.numeric(st_area(.))) %>%
    mutate(pop_density = POP_EST / area)

ggplot(europe_map_data) + geom_sf(aes(fill=pop_density)) +
    theme_minimal() + 
    scale_fill_continuous_tableau(palette = "Green")
```

![]({{ site.baseurl }}/images/map_article_2018_files/figure-markdown_strict/pop_density-1.png)

Using aggregating functions of the `tidyverse` package is also
straight-forward. Lets create a similar population density plot but
instead for each subregion of Europe.

```R
subregion_data <- europe_map_data %>%
    group_by(SUBREGION) %>%
    summarise(area = sum(area), 
            pop_est = sum(POP_EST)) %>%
    ungroup() %>%
    mutate(pop_density = pop_est / area)

ggplot(subregion_data) + geom_sf(aes(fill=pop_density)) +
    theme_minimal() + 
    scale_fill_continuous_tableau(palette = "Green")
```

![]({{ site.baseurl }}/images/map_article_2018_files/figure-markdown_strict/pop_density_sub-1.png)

As a last exercise lets find the centroid for each country.

```R
# First get all centroids of each European country
get_coordinates = function(data) {
    return_data <- data %>%
    st_geometry() %>%
    st_centroid() %>%
    st_coordinates() %>%
    as_data_frame()
}

europe_centres <- europe_map_data %>%
    group_by(NAME) %>%
    do(get_coordinates(.))

europe_map_data <- europe_map_data %>%
    left_join(europe_centres, by="NAME")
```

Actually, I only want to see the centroid of the Netherlands...

```R
netherlands_map_data = europe_map_data %>%
    filter(NAME == "Netherlands") %>%
    st_crop(xmin=1, xmax=10, ymin=50, ymax=55)

## although coordinates are longitude/latitude, st_intersection assumes that they are planar

## Warning: attribute variables are assumed to be spatially constant
## throughout all geometries

ggplot(netherlands_map_data) + geom_sf() +
    geom_point(aes(x=X, y=Y, colour="red")) + 
    theme_minimal()
```

![]({{ site.baseurl }}/images/map_article_2018_files/figure-markdown_strict/centroid_netherlands-1.png)

## Setup

The analysis of this tutorial is performed using R version 3.5.1. To use
the `st_crop` function from the `sf` package version 0.6.3 is needed.
`geom_sf` also requires a recent version of `ggplot2`.
