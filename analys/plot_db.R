library(DBI)
library(RSQLite)
library(plot3D)

#con = dbConnect(SQLite(), dbname="hugo2_7.sqlite3")
con = dbConnect(SQLite(), dbname="big_data_10M.sqlite3")

myQuery <- dbSendQuery(con, "SELECT mfv, heading FROM kinetics WHERE 9234000 < seq_id AND seq_id < 9239000")
data <- dbFetch(myQuery, n = -1)

## Polar plot

x <- cos(data$heading)*data$mfv
y <- sin(data$heading)*data$mfv

# Polar plot
plot(x,y,xlim=c(-1.2, 1.2), ylim=c(-1.2, 1.2))


##  Create cuts:
x_c <- cut(x, 30)
y_c <- cut(y, 30)

##  Calculate joint counts at cut levels:
z <- table(x_c, y_c)

##  Plot as a 2D heatmap:
#image2D(z=z, border="black")

##  Plot as a 3D histogram:
#hist3D(z=z, border="black")

# in average heading should be pi (3.14) ideally
#myQuery <- dbSendQuery(con, "SELECT avg(heading) FROM kinetics")
#quality_1 <- dbFetch(myQuery, n = -1)

# in average mfv should be 1.00 ideally
#myQuery <- dbSendQuery(con, "SELECT avg(mfv) FROM kinetics")
#quality_2 <- dbFetch(myQuery, n = -1)

dbClearResult(myQuery)
