
#include <stdio.h>
#include <stdint.h>
#include "math.h"

// point_square calculates square error of a set of points (mxt,myt) against a model (x0, y0, scale_r, scale_y_axis, theta)
// it returns ...
// mxt  magnetometer x
// myt  magnetometer y
// x0 origo in model
// x1 origo in model
// scale_r in model; scale to unity circle with r=1
// scale_y_axis in model; scale the rotated ellipse to a circle
// theta in model (radians 0) <= theta <= pi/2
void point_square(float mxt, float myt, float x0, float y0, float scale_r, float scale_y_axis, float theta,
                  float *normalized_x, float *normalized_y, float *quad_error) {

    //printf("raw,%f,%f\n", x, y);

    float trans_x = (mxt - x0);  // move plane to set origo to in middle of ellipse
    float trans_y = (myt - y0);  // move plane to set origo to in middle of ellipse

    //printf("trans,%f,%f\n", trans_x, trans_y);

    // Vector is individual for each rotated pair (mx,my)
    // unless they are situated on a perfect circle which
    // measured values are NOT!! (otherwise v would be r)

    // https://en.wikipedia.org/wiki/Rotation_(mathematics)
    // https://en.wikipedia.org/wiki/C_mathematical_functions

    float rotate_x = trans_x * cosf(theta) - trans_y * sinf(theta);  // Rotate theta degrees
    float rotate_y = trans_x * sinf(theta) + trans_y * cosf(theta);  // Rotate theta degrees

    //printf("rotate,%f,%f\n", rotate_x, rotate_y);

    float scale_x = rotate_x;
    float scale_y = rotate_y * scale_y_axis;                         // Scale y-axis to make the ellips to a cicle

    //printf("scale,%f,%f\n", scale_x, scale_y); // För R-plot

    rotate_x = scale_x * cosf(-theta) - scale_y * sinf(-theta);      // Rotate -theta degrees (back)
    rotate_y = scale_x * sinf(-theta) + scale_y * cosf(-theta);      // Rotate -theta degrees (back)

    //printf("rotate back ,%f,%f\n", rotate_x, rotate_y);

    *normalized_x = rotate_x * scale_r;                        // Returns x,y normalized to unit circle
    *normalized_y = rotate_y * scale_r;

    //printf("normalized ,%f,%f\n", *normalized_x, *normalized_y);

    *quad_error = powf(sqrtf(powf(*normalized_x,2) + powf(*normalized_y,2)) - 1,2); // Returns square error from unity cicle

    //printf("quad_error,%f\n", quad_error);

}
