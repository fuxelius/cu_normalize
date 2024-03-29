library(plot3D)

##  Simulate data:
#set.seed(2002)
#x <- rnorm(1000)
#y <- rnorm(1000)

x <- apa$x
y <- apa$y

#x <- cos(data$heading)*data$mfv
#y <- sin(data$heading)*data$mfv

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

## Polar plot
#plot(cos(apa2$rho)*apa2$mfv,sin(apa2$rho)*apa2$mfv,xlim=c(-1.2, 1.2), ylim=c(-1.2, 1.2))