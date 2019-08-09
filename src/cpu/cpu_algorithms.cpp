//
// Created by michele on 07/07/19.
//

#include <sensor_msgs/Image.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include "cpu_algorithms.h"
#include "../utilities/utilities.h"

CpuAlgorithms::CpuAlgorithms() {}

CpuAlgorithms& CpuAlgorithms::getInstance() {
    static CpuAlgorithms cpuAlgorithms;
    return cpuAlgorithms;
}

void CpuAlgorithms::toGrayScale(unsigned char *pixels, int width, int height) {
    int len = width * height * 3;
    for (int i = 0; i < len; i += 3) {
        unsigned char average = ( *(pixels) + *(pixels + 1) + *(pixels + 2) ) / 3;
        *(pixels++) = average;
        *(pixels++) = average;
        *(pixels++) = average;
    }
}

void CpuAlgorithms::toGrayScale(uint8_t *destination, const uint8_t *source, int width, int height){
    int len = width * height * 3;

    for (int i = 0; i < len; i += 3) {
        unsigned char average = ( *(source++) + *(source++) + *(source++) ) / 3;
        *(destination++) = average;
        *(destination++) = average;
        *(destination++) = average;
    }
}

void CpuAlgorithms::copyArrayToImage(sensor_msgs::Image& destination, uint8_t *source){
    int len = destination.width * destination.height * 3;

    for (int i = 0; i < len; i += 3) {
        destination.data.push_back(*(source++));
        destination.data.push_back(*(source++));
        destination.data.push_back(*(source++));
    }
}

void CpuAlgorithms::gaussianFilter(unsigned char *destination, unsigned char *source, float *matrix, int width,
                                   int height, int matrixDim) {

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = 0;
            for (int k = -matrixDim / 2; k <= matrixDim / 2; ++k) {
                for (int z = -matrixDim / 2; z <= matrixDim / 2; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        value += *(source + ((i + k) * width + j + z) * 3) * matrix[(k + matrixDim / 2) * matrixDim + (z + matrixDim / 2)];
                    }
                }
            }
            *(destination + (i * width + j) * 3) = value;
            *(destination + (i * width + j) * 3 + 1) = value;
            *(destination + (i * width + j) * 3 + 2)  = value;
        }
    }
}

void CpuAlgorithms::sobel(float *edgeGradient, int *edgeDirection, unsigned char *source, int width, int height){

    int maskDim = 3;
    int *sobelMaskHorizontal = Utilities::getSobelMaskHorizontal();

    float *sobelHorizontal = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = 0;
            for (int k = -maskDim / 2; k <= maskDim / 2; ++k) {
                for (int z = -maskDim / 2; z <= maskDim / 2; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        value += *(source + ((i + k) * width + j + z) * 3) * sobelMaskHorizontal[(k + maskDim / 2) * maskDim + (z + maskDim / 2)];
                    }
                }
            }

            *(sobelHorizontal + i * width + j) = value;
        }
    }

    int *sobelMaskVertical = Utilities::getSobelMaskVertical();
    float *sobelVertical = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = 0;
            for (int k = -maskDim / 2; k <= maskDim / 2; ++k) {
                for (int z = -maskDim / 2; z <= maskDim / 2; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        value += *(source + ((i + k) * width + j + z) * 3) * sobelMaskVertical[(k + maskDim / 2) * maskDim + (z + maskDim / 2)];
                    }
                }
            }

            *(sobelVertical + i * width + j) = value;
        }
    }

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = sqrt(pow(sobelHorizontal[i * width + j], 2) + pow(sobelVertical[i * width + j], 2));
            *(edgeGradient + i * width + j) = value;
            float dir = atan2(sobelVertical[i * width + j], sobelHorizontal[i * width + j]) * 180 / M_PI;
            if (dir < 0)
                dir += 180;
            if(dir > 22.5 && dir <= 67.5)
                dir = 45;
            else if(dir > 67.5 && dir <= 112.5)
                dir = 90;
            else if(dir > 112.5 && dir <= 157.5)
                dir = 135;
            else
                dir = 0;

            *(edgeDirection + i * width + j) = dir;
        }
    }

    delete(sobelMaskHorizontal);
    delete(sobelMaskVertical);
    delete(sobelHorizontal);
    delete(sobelVertical);
}


void CpuAlgorithms::nonMaximumSuppression(unsigned char *destination, float *edgeGradient, int *edgeDirection, int width, int height) {

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int dir = *(edgeDirection + i * width + j);
            float first = 0;
            float second = 0;

            if(dir == 0){
                if(j - 1 >= 0)
                    first = *(edgeGradient + i * width + j - 1);
                if(j + 1 < width)
                    second = *(edgeGradient + i * width + j + 1);
            }
            else if(dir == 90){
                if(i - 1 >= 0)
                    first = *(edgeGradient + (i - 1) * width + j);
                if(i + 1 < height)
                    second = *(edgeGradient + (i + 1) * width + j);
            }
            else if(dir == 45){
                if(i + 1 < height && j + 1 < width)
                    first = *(edgeGradient + (i + 1) * width + j + 1);
                if(i - 1 >= 0 && j - 1 >= 0)
                    second = *(edgeGradient + (i - 1) * width + j - 1);

            }
            else if(dir == 135){
                if(i - 1 >= 0 && j + 1 < width)
                    first = *(edgeGradient + (i - 1) * width + j + 1);
                if(i + 1 < height && j - 1 >= 0)
                    second = *(edgeGradient + (i + 1) * width + j - 1);
            }

            float currentValue = *(edgeGradient + i * width + j);

            if(!(currentValue >= first && currentValue >= second))
                currentValue = 0;
            else if(currentValue > 50)
                currentValue = 255;
            else
                currentValue = 0;

            *(destination + (i * width + j) * 3) = currentValue;
            *(destination + (i * width + j) * 3 + 1) = currentValue;
            *(destination + (i * width + j) * 3 + 2)  = currentValue;
        }
    }
}

void CpuAlgorithms::harris(unsigned char *destination, unsigned char *source, unsigned char *imageSobel, int width, int height) {
    int maskDim = 3;
    int *sobelMaskHorizontal = Utilities::getSobelMaskHorizontal();

    float *sobelHorizontal = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = 0;
            for (int k = -maskDim / 2; k <= maskDim / 2; ++k) {
                for (int z = -maskDim / 2; z <= maskDim / 2; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        value += *(source + ((i + k) * width + j + z) * 3) * sobelMaskHorizontal[(k + maskDim / 2) * maskDim + (z + maskDim / 2)];
                    }
                }
            }

            *(sobelHorizontal + i * width + j) = value;
        }
    }

    int *sobelMaskVertical = Utilities::getSobelMaskVertical();
    float *sobelVertical = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float value = 0;
            for (int k = -maskDim / 2; k <= maskDim / 2; ++k) {
                for (int z = -maskDim / 2; z <= maskDim / 2; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        value += *(source + ((i + k) * width + j + z) * 3) * sobelMaskVertical[(k + maskDim / 2) * maskDim + (z + maskDim / 2)];
                    }
                }
            }

            *(sobelVertical + i * width + j) = value;
        }
    }

    double time = Utilities::seconds();

    float *sobelHorizontal2 = new float[width * height];
    float *sobelVertical2 = new float[width * height];
    float *sobelHorizontalVertical = new float[width * height];

    // corner detection
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float x = *(sobelHorizontal + i * width + j);
            float y = *(sobelVertical + i * width + j);

            *(sobelHorizontal2 + i * width + j) = x * x;
            *(sobelVertical2 + i * width + j) = y * y;
            *(sobelHorizontalVertical + i * width + j) = x * y;
        }
    }

    float *sobelHorizontal2Sum = new float[width * height];
    float *sobelVertical2Sum = new float[width * height];
    float *sobelHorizontalVerticalSum = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float valueX = 0;
            float valueY = 0;
            float valueXY = 0;
            for (int k = -1; k <= 1; ++k) {
                for (int z = -1; z <= 1; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        valueX += *(sobelHorizontal2 + ((i + k) * width + j + z));
                        valueY += *(sobelVertical2 + ((i + k) * width + j + z));
                        valueXY += *(sobelHorizontalVertical + ((i + k) * width + j + z));
                    }
                }
            }

            *(sobelHorizontal2Sum + i * width + j) = valueX;
            *(sobelVertical2Sum + i * width + j) = valueY;
            *(sobelHorizontalVerticalSum + i * width + j) = valueXY;
        }
    }

    float *corners = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float x = *(sobelHorizontal2Sum + i * width + j);
            float y = *(sobelVertical2Sum + i * width + j);
            float xy = *(sobelHorizontalVerticalSum + i * width + j);

            *(corners + i * width + j) = (x * y - xy * xy) - 0.06f * ((x + y) * (x + y));
        }
    }

    float *cornerSuppressed = new float[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float max = true;
            float currentValue = *(corners + i * width + j);

            for (int k = -1; k <= 1 && max; ++k) {
                for (int z = -1; z <= 1 && max; ++z) {
                    if(i + k >= 0 && i + k < height &&
                       j + z >= 0 && j + z < width) {
                        if(currentValue < *(corners + ((i + k) * width + j + z)))
                            max = false;
                    }
                }
            }

            if(currentValue > 9000000){
                if(!max)
                    currentValue = 0;
                else
                    currentValue = 255;
            } else
                currentValue = 0;

            *(cornerSuppressed + (i * width + j)) = currentValue;
        }
    }

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float r = *(cornerSuppressed + i * width + j);
            unsigned  char final;
            if(r == 0){
                final = *(imageSobel + (i * width + j)*3);
                *(destination + (i * width + j) * 3) = final;
                *(destination + (i * width + j) * 3 + 1) = final;
                *(destination + (i * width + j) * 3 + 2) = final;
            }

            else{
                *(destination + (i * width + j) * 3) = 0;
                *(destination + (i * width + j) * 3 + 1) = 255;
                *(destination + (i * width + j) * 3 + 2) = 0;
            }

        }
    }

    delete(sobelMaskHorizontal);
    delete(sobelMaskVertical);
    delete(sobelHorizontal);
    delete(sobelVertical);
    delete(cornerSuppressed);
    delete(corners);
    delete(sobelHorizontal2);
    delete(sobelHorizontal2Sum);
    delete(sobelVertical2);
    delete(sobelVertical2Sum);
    delete(sobelHorizontalVertical);
    delete(sobelHorizontalVerticalSum);
}

double CpuAlgorithms::houghLinesIntersection(vector<Point> &points, Mat &image) {
    vector<Vec2f> lines;

    double time = Utilities::seconds();

    HoughLines(image, lines, 1, CV_PI/180, 110, 0, 0 );
    for (int i = 0; i < lines.size() - 1; ++i) {
        for (int j = i + 1; j < lines.size(); ++j) {
            float rho1 = lines[i][0], theta1 = lines[i][1], rho2 = lines[j][0], theta2 = lines[j][1];
            Point pt1, pt2, pt3, pt4;
            double a = cos(theta1), b = sin(theta1), c = cos(theta2), d = sin(theta2);
            double x1 = a*rho1, y1 = b*rho1, x2 = c*rho2, y2 = d*rho2;
            pt1.x = cvRound(x1 + 1000*(-b));
            pt1.y = cvRound(y1 + 1000*(a));
            pt2.x = cvRound(x1 - 1000*(-b));
            pt2.y = cvRound(y1 - 1000*(a));

            pt3.x = cvRound(x2 + 1000*(-d));
            pt3.y = cvRound(y2 + 1000*(c));
            pt4.x = cvRound(x2 - 1000*(-d));
            pt4.y = cvRound(y2 - 1000*(c));

            Point x = pt3 - pt1;
            Point d1 = pt2 - pt1;
            Point d2 = pt4 - pt3;

            float cross = d1.x * d2.y - d1.y * d2.x;
            if (abs(cross) >= 1e-8f){
                Point r;
                double t1 = (x.x * d2.y - x.y * d2.x) / cross;
                r = pt1 + d1 * t1;
                if(r.x >= 0 && r.x < image.cols && r.y >= 0 && r.y < image.rows){
                    points.push_back(r);
                }
            }
        }
    }

    return Utilities::seconds() - time;
}

double CpuAlgorithms::findCandidateCorner(vector<Point> &candidateCorners, unsigned char *cornerImage, vector<Point> &intersections, int width, int height) {

    unsigned char B;
    unsigned char G;
    unsigned char R;
    Point point;
    int mask = 5;
    double time = Utilities::seconds();
    for(int i = 0; i < intersections.size(); ++i){
        point = intersections[i];
        for(int y = point.y - mask / 2; y <= point.y + mask / 2; y++){
            if (y >= 0 && y < height){
                for(int x = point.x - mask / 2; x <= point.x + mask / 2; x++){
                    if (x >= 0 && x < width){
                        B = *(cornerImage + (y * width + x) * 3);
                        G = *(cornerImage + (y * width + x) * 3 + 1);
                        R = *(cornerImage + (y * width + x) * 3 + 2);
                        if(B == 0 && G == 255 && R == 0){
                            candidateCorners.push_back(Point(x, y));
                            *(cornerImage + (y * width + x) * 3) = 0;
                            *(cornerImage + (y * width + x) * 3 + 1) = 0;
                            *(cornerImage + (y * width + x) * 3 + 2) = 0;
                        }
                    }
                }
            }
        }
    }
    return Utilities::seconds() - time;
}

void drawLines(Mat *image, Point a, Point b, Point c, Point d){
    image->setTo(0);

    line(*image, a, b, Scalar(0, 0, 255), 4);
    line(*image, b, c, Scalar(0, 0, 255), 4);

    line(*image, c, d, Scalar(0, 0, 255), 4);

    line(*image, d, a, Scalar(0, 0, 255), 4);
}

double CpuAlgorithms::candidateGroups(vector<pair<vector<Point>, Mat*>> &groups, vector<Point> &corners, Mat &image, int width, int height) {

    float diagonal = sqrt(width * width + height * height);
    float heightThresL = 0.5;
    float heightThresH = 0.9;
    float widthThresH = 0.8;
    float widthThresL = 0.1;
    float directionThresL = 15;
    float directionThresH = 87;
    float parallelThres = 1.5;
    float ratioThresL = 2.0;
    float ratioThresH = 3.0;

    vector<thread> threads;

    double time = Utilities::seconds();
    for (int i = 0; i < corners.size(); ++i) {
        for (int y = 0; y < corners.size(); ++y) {
            Point a = corners[i];
            Point b = corners[y];
            float Sab = sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2)) / diagonal;
            float Dab = atan(abs((1.0 * a.x - b.x)) / abs(1.0 * a.y - b.y)) * (180 / (float) M_PI);

            if(i != y && a.x < b.x && widthThresL < Sab && Sab < widthThresH && Dab > directionThresH) {
                for (int z = 0; z < corners.size(); ++z) {
                    Point c = corners[z];
                    float Sbc = sqrt(pow(c.x - b.x, 2) + pow(c.y - b.y, 2)) / diagonal;
                    float Dbc = atan(abs((1.0 * c.x - b.x)) / abs(1.0 * c.y - b.y)) * (180 / (float) M_PI);

                    if(y != z && b.y < c.y && a.x < c.x && a.y < c.y && heightThresL < Sbc && Sbc < heightThresH && Dbc < directionThresL) {
                        for (int t = 0; t < corners.size(); ++t) {
                            Point d = corners[t];

                            float Scd = sqrt(pow(c.x - d.x, 2) + pow(c.y - d.y, 2)) / diagonal;
                            float Dcd = atan(abs((1.0 * c.x - d.x)) / abs(1.0 * c.y - d.y)) * (180 / (float) M_PI);

                            float Sda = sqrt(pow(d.x - a.x, 2) + pow(d.y - a.y, 2)) / diagonal;
                            float Dda = atan(abs((1.0 * d.x - a.x)) / abs(1.0 * d.y - a.y)) * (180 / (float) M_PI);
                            //printf("%f, %f\n", Sda, Dda);
                            if (z != t && d.x < c.x && d.x < b.x && b.y < d.y && a.y < d.y && widthThresL < Scd && Scd < widthThresH && Dcd > directionThresH &&
                            heightThresL < Sda && Sda < heightThresH && Dda < directionThresL &
                            abs(Dda - Dbc) < parallelThres && ratioThresL < (Sda + Sbc) / (Scd + Sab) &&
                            (Sda + Sbc) / (Scd + Sab) < ratioThresH) {

                                vector<Point> group;
                                group.push_back(a);
                                group.push_back(b);
                                group.push_back(c);
                                group.push_back(d);

                                bool found = false;
                                for (int j = 0; j < groups.size() && !found; ++j) {
                                    vector<Point> oldGroup = groups[j].first;
                                    Point p1 = group[0] - oldGroup[0];
                                    Point p2 = group[1] - oldGroup[1];
                                    Point p3 = group[2] - oldGroup[2];
                                    Point p4 = group[3] - oldGroup[3];
                                    if(abs(p1.x) < 5 && abs(p1.y) < 5 && abs(p2.x) < 5 && abs(p2.y) < 5 && abs(p3.x) < 5 &&
                                            abs(p3.y) < 5 && abs(p4.x) < 5 && abs(p4.y) < 5)
                                        found = true;
                                }

                                if(!found) {
                                    Mat *poly = new Mat(height, width, CV_8UC3);
                                    groups.push_back(pair<vector<Point>, Mat*>(group, poly));
                                    //printf("a: %i, %i, b: %i, %i, c: %i, %i, d: %i, %i\n", a.x, a.y, b.x, b.y, c.x, c.y,
                                      //     d.x, d.y);
                                    threads.push_back(thread(drawLines, poly, a, b, c, d));
                                    /*Mat im(height, width, CV_8UC3);
                                    for (int j = 0; j < width * height * 3; ++j) {
                                        im.data[j] = image.data[j];
                                    }
                                    line(im, a, b, Scalar(0, 0, 255), 4);
                                    line(im, b, c, Scalar(0, 0, 255), 4);

                                    line(im, c, d, Scalar(0, 0, 255), 4);

                                    line(im, d, a, Scalar(0, 0, 255), 4);
                                    imshow("ciao", im);
                                    waitKey(0);
                                    */
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < threads.size(); ++i) {
        threads[i].join();
        //imshow("ciao", *(groups[i].second));
        //waitKey(0);
    }
    return Utilities::seconds() - time;
}

double CpuAlgorithms::fillRatio(vector<vector<Point>>& matchFillRatio, vector<pair<vector<Point>, Mat *>> &groups, unsigned char *image, int width, int height) {

    float FRThresL = 0.6;
    float FRThrefH = 0.85;

    double time = Utilities::seconds();

    for (int i = 0; i < groups.size(); ++i) {
        vector<Point> group = groups[i].first;
        Mat *poly = groups[i].second;
        //imshow("ciao", *poly);
        //waitKey(0);

        // AB line
        int x = group[0].x, y = group[0].y;
        int lenAB = 0, overlapAB = 0;
        bool dir = false;
        if (y < group[1].y)
            dir = true;
        while(x < group[1].x) {
            bool foundNext = false;
            int maskX = 1, maskY = 1;
            while (!foundNext) {
                if (dir && x + maskX < width) {
                    for (int j = -maskY; j <= maskY && !foundNext; ++j) {
                        if (y + j >= 0 && y + j < height && image[((y + j) * width + x + maskX) * 3] == 255) {
                            foundNext = true;
                            x += maskX;
                            y += j;
                            lenAB++;
                            if (poly->data[((y + j) * width + x + maskX) * 3 + 2] == 255)
                                overlapAB++;
                        }
                    }
                } else if (!dir && x + maskX < width) {
                    for (int j = maskY; j >= -maskY && !foundNext; --j) {
                        if (y + j >= 0 && y + j < height && image[((y + j) * width + x + maskX) * 3] == 255) {
                            foundNext = true;
                            x += maskX;
                            y += j;
                            lenAB++;
                            if (poly->data[((y + j) * width + x + maskX) * 3 + 2] == 255)
                                overlapAB++;
                        }
                    }
                }
                maskX++;
                maskY++;
            }
        }


        // BC line
        x = group[1].x, y = group[1].y;
        int lenBC = 0, overlapBC = 0;
        dir = false;
        if (x < group[2].x)
            dir = true;
        while (y < group[2].y) {
            bool foundNext = false;
            int maskX = 1, maskY = 1;
            while (!foundNext) {
                if (dir && y + maskY < height) {
                    for (int j = maskX; j >= -maskX && !foundNext; --j) {
                        if (x + j >= 0 && x + j < width &&
                        image[((y + maskY) * width + x + j) * 3] == 255) {
                            foundNext = true;
                            x += j;
                            y += maskY;
                            lenBC++;
                            if (poly->data[((y + maskY) * width + x + j) * 3 + 2] == 255)
                                overlapBC++;
                        }
                    }
                } else if (!dir && y + maskY < height) {
                    for (int j = -maskX; j <= maskX && !foundNext; ++j) {
                        if (x + j >= 0 && x + j < width &&
                        image[((y + maskY) * width + x + j) * 3] == 255) {
                            foundNext = true;
                            x += j;
                            y += maskY;
                            lenBC++;
                            if (poly->data[((y + maskY) * width + x + j) * 3 + 2] == 255)
                                overlapBC++;
                        }
                    }
                }
                maskX++;
                maskY++;
            }
        }

        // DC line ->
        x = group[3].x, y = group[3].y;
        int lenDC = 0, overlapDC = 0;
        dir = false;
        if (y < group[2].y)
            dir = true;
        while(x < group[2].x) {
            bool foundNext = false;
            int maskX = 1, maskY = 1;
            while (!foundNext) {
                if (dir && x + maskX < width) {
                    for (int j = -maskY; j <= maskY && !foundNext; ++j) {
                        if (y + j >= 0 && y + j < height && image[((y + j) * width + x + maskX) * 3] == 255) {
                            foundNext = true;
                            x += maskX;
                            y += j;
                            lenDC++;
                            if (poly->data[((y + j) * width + x + maskX) * 3 + 2] == 255)
                                overlapDC++;
                        }
                    }
                } else if (!dir && x + maskX < width) {
                    for (int j = maskY; j >= -maskY && !foundNext; --j) {
                        if (y + j >= 0 && y + j < height && image[((y + j) * width + x + maskX) * 3] == 255) {
                            foundNext = true;
                            x += maskX;
                            y += j;
                            lenDC++;
                            if (poly->data[((y + j) * width + x + maskX) * 3 + 2] == 255)
                                overlapDC++;
                        }
                    }
                }
                maskX++;
                maskY++;
            }
        }


        // AD line
        x = group[0].x, y = group[0].y;
        int lenAD = 0, overlapAD = 0;
        dir = false;
        if (x < group[3].x)
            dir = true;
        while (y < group[3].y) {
            bool foundNext = false;
            int maskX = 1, maskY = 1;
            while (!foundNext) {
                if (dir && y + maskY < height) {
                    for (int j = maskX; j >= -maskX && !foundNext; --j) {
                        if (x + j >= 0 && x + j < width &&
                            image[((y + maskY) * width + x + j) * 3] == 255) {
                            foundNext = true;
                            x += j;
                            y += maskY;
                            lenAD++;
                            if (poly->data[((y + maskY) * width + x + j) * 3 + 2] == 255)
                                overlapAD++;
                        }
                    }
                } else if (!dir && y + maskY < height) {
                    for (int j = -maskX; j <= maskX && !foundNext; ++j) {
                        if (x + j >= 0 && x + j < width &&
                            image[((y + maskY) * width + x + j) * 3] == 255) {
                            foundNext = true;
                            x += j;
                            y += maskY;
                            lenAD++;
                            if (poly->data[((y + maskY) * width + x + j) * 3 + 2] == 255)
                                overlapAD++;
                        }
                    }
                }
                maskX++;
                maskY++;
            }
        }

        //printf("%i %i %i %i %i %i %i %i\n", lenAB, overlapAB, lenBC, overlapBC, lenDC, overlapDC, lenAD, overlapAD);

        float frAB = overlapAB * 1.0f / lenAB;
        float frBC = overlapBC * 1.0f / lenBC;
        float frAD = overlapAD * 1.0f / lenAD;
        float frDC = overlapDC * 1.0f / lenDC;

        if(frAB >= FRThresL && frBC >= FRThresL && frDC >= FRThresL && frAD >= FRThresL){
            matchFillRatio.push_back(group);
            //printf("trovato!\n");
        }

    }

    return Utilities::seconds() - time;

}
