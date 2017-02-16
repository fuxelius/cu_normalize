library(DBI)
library(RSQLite)
con = dbConnect(SQLite(), dbname="hugo2_7.sqlite3")
myQuery <- dbSendQuery(con, "SELECT mfv, heading FROM kinetics WHERE 28068 < seq_id AND seq_id < 29159")
data <- dbFetch(myQuery, n = -1)

## Polar plot
plot(cos(data$heading)*data$mfv,sin(data$heading)*data$mfv,xlim=c(-1.2, 1.2), ylim=c(-1.2, 1.2))


# in average heading should be pi (3.14) ideally
myQuery <- dbSendQuery(con, "SELECT avg(heading) FROM kinetics")
quality_1 <- dbFetch(myQuery, n = -1)

# in average mfv should be 1.00 ideally
myQuery <- dbSendQuery(con, "SELECT avg(mfv) FROM kinetics")
quality_2 <- dbFetch(myQuery, n = -1)

dbClearResult(myQuery)
