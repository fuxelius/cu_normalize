library(plot3D)

##  Simulate data:
#set.seed(2002)
#x <- rnorm(1000)
#y <- rnorm(1000)

x <- apa$x
y <- apa$y

##  Create cuts:
x_c <- cut(x, 30)
y_c <- cut(y, 30)



##  Calculate joint counts at cut levels:
z <- table(x_c, y_c)

plot(x,y)

##  Plot as a 2D heatmap:
#image2D(z=z, border="black")

##  Plot as a 3D histogram:
#hist3D(z=z, border="black")

