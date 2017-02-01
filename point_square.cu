
#include <stdio.h>
#include <stdint.h>
#include "math.h"


float point_square(float mxt, float myt, float x0, float y0, float scale_r, float scale, float theta) {

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
    float scale_y = rotate_y * scale;                                // Scale y-axis

    //printf("scale,%f,%f\n", scale_x, scale_y); // För R-plot

    rotate_x = scale_x * cosf(-theta) - scale_y * sinf(-theta);  // Rotate -theta degrees
    rotate_y = scale_x * sinf(-theta) + scale_y * cosf(-theta);  // Rotate -theta degrees

    //printf("rotate back ,%f,%f\n", rotate_x, rotate_y);

    float normalized_x = rotate_x * scale_r;
    float normalized_y = rotate_y * scale_r;

    printf("normalized ,%f,%f\n", normalized_x, normalized_y);

    float quad_error = powf(sqrtf(powf(normalized_x,2) + powf(normalized_y,2)) - 1,2);

    //printf("quad_error,%f\n", quad_error);

    return (quad_error);

}
