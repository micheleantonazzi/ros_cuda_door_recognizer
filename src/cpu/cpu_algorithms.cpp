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

            if(currentValue > 90000){
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
            double x1 = a * rho1, y1 = b * rho1, x2 = c * rho2, y2 = d * rho2;
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
                Point intersection;
                double t1 = (x.x * d2.y - x.y * d2.x) / cross;
                intersection = pt1 + d1 * t1;
                if(intersection.x >= 0 && intersection.x < image.cols && intersection.y >= 0 && intersection.y < image.rows){
                    points.push_back(intersection);
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

    line(*image, a, b, Scalar(0, 0, 255), 3);
    line(*image, b, c, Scalar(0, 0, 255), 3);

    line(*image, c, d, Scalar(0, 0, 255), 3);

    line(*image, d, a, Scalar(0, 0, 255), 3);
}

double CpuAlgorithms::candidateGroups(vector<pair<vector<Point>, Mat*>> &groups, vector<Point> &corners, Mat &image, int width, int height) {

    float diagonal = sqrt(width * width + height * height);
    float heightL = 0.5;
    float heightH = 0.9;
    float widthH = 0.8;
    float widthL = 0.1;
    float directionL = 15;
    float directionH = 85;
    float parallel = 1.5;
    float ratioL = 2.0;
    float ratioH = 3.0;

    vector<thread> threads;

    double time = Utilities::seconds();
    for (int i = 0; i < corners.size(); ++i) {
        for (int y = 0; y < corners.size(); ++y) {
            Point C1 = corners[i];
            Point C2 = corners[y];
            float SC1C2 = sqrt(pow(C1.x - C2.x, 2) + pow(C1.y - C2.y, 2)) / diagonal;
            float DC1C2 = atan(abs((1.0 * C1.x - C2.x)) / abs(1.0 * C1.y - C2.y)) * (180 / (float) M_PI);

            if(i != y && C1.x < C2.x && widthL < SC1C2 && SC1C2 < widthH && DC1C2 > directionH) {
                for (int z = 0; z < corners.size(); ++z) {
                    Point C3 = corners[z];
                    float SC2C3 = sqrt(pow(C3.x - C2.x, 2) + pow(C3.y - C2.y, 2)) / diagonal;
                    float DC2C3 = atan(abs((1.0 * C3.x - C2.x)) / abs(1.0 * C3.y - C2.y)) * (180 / (float) M_PI);

                    if(y != z && C2.y < C3.y && C1.x < C3.x && C1.y < C3.y && heightL < SC2C3 && SC2C3 < heightH && DC2C3 < directionL) {
                        for (int t = 0; t < corners.size(); ++t) {
                            Point C4 = corners[t];

                            float SC3C4 = sqrt(pow(C3.x - C4.x, 2) + pow(C3.y - C4.y, 2)) / diagonal;
                            float DC3C4 = atan(abs((1.0 * C3.x - C4.x)) / abs(1.0 * C3.y - C4.y)) * (180 / (float) M_PI);

                            float SC4C1 = sqrt(pow(C4.x - C1.x, 2) + pow(C4.y - C1.y, 2)) / diagonal;
                            float DC4C1 = atan(abs((1.0 * C4.x - C1.x)) / abs(1.0 * C4.y - C1.y)) * (180 / (float) M_PI);
                            //printf("%f, %f\n", SC4C1, DC4C1);
                            if (z != t && C4.x < C3.x && C4.x < C2.x && C2.y < C4.y && C1.y < C4.y && widthL < SC3C4 && SC3C4 < widthH && DC3C4 > directionH &&
                                heightL < SC4C1 && SC4C1 < heightH && DC4C1 < directionL &
                                                                      abs(DC4C1 - DC2C3) < parallel && ratioL < (SC4C1 + SC2C3) / (SC3C4 + SC1C2) &&
                                (SC4C1 + SC2C3) / (SC3C4 + SC1C2) < ratioH) {

                                vector<Point> group;
                                group.push_back(C1);
                                group.push_back(C2);
                                group.push_back(C3);
                                group.push_back(C4);

                                bool found = false;
                                for (int j = 0; j < groups.size() && !found; ++j) {
                                    vector<Point> oldGroup = groups[j].first;
                                    Point p1 = group[0] - oldGroup[0];
                                    Point p2 = group[1] - oldGroup[1];
                                    Point p3 = group[2] - oldGroup[2];
                                    Point p4 = group[3] - oldGroup[3];
                                    if((abs(p1.x) < 5 && abs(p1.y) < 5 && abs(p2.x) < 5 && abs(p2.y) < 5 && abs(p3.x) < 5 &&
                                            abs(p3.y) < 5 && abs(p4.x) < 5 && abs(p4.y) < 5))
                                        found = true;
                                }

                                if(!found) {
                                    Mat *poly = new Mat(height, width, CV_8UC3);
                                    groups.push_back(pair<vector<Point>, Mat*>(group, poly));
                                    //printf("C1: %i, %i, C2: %i, %i, C3: %i, %i, C4: %i, %i\n", C1.x, C1.y, C2.x, C2.y, C3.x, C3.y,
                                      //     C4.x, C4.y);
                                    threads.push_back(thread(drawLines, poly, C1, C2, C3, C4));
                                    /*Mat im(height, width, CV_8UC3);
                                    for (int j = 0; j < width * height * 3; ++j) {
                                        im.data[j] = image.data[j];
                                    }
                                    line(im, C1, C2, Scalar(0, 0, 255), 4);
                                    line(im, C2, C3, Scalar(0, 0, 255), 4);

                                    line(im, C3, C4, Scalar(0, 0, 255), 4);

                                    line(im, C4, C1, Scalar(0, 0, 255), 4);
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

    float fillRatioLOne = 0.6;
    float fillRatioHOne = 0.85;

    float fillRatioLTwo = 0.87;
    float fillRatioHTwo = 0.9;

    vector<vector<Point>> matchFillRatioOne;
    vector<vector<Point>> matchFillRatioTwo;

    double time = Utilities::seconds();

    for (int i = 0; i < groups.size(); ++i) {
        vector<Point> group = groups[i].first;
        Mat *poly = groups[i].second;
        //imshow("ciao", *poly);
        //waitKey(0);

        // L12
        int x = group[0].x, y = group[0].y;
        int len12 = 0, overlap12 = 0;
        bool dir = false;
        if (y < group[1].y)
            dir = true;
        while(x <= group[1].x) {
            bool foundNext = false;
            int maskX = 1, maskY = 1;
            while (!foundNext) {
                if (dir && x + maskX < width) {
                    for (int j = -maskY; j <= maskY && !foundNext; ++j) {
                        if (y + j >= 0 && y + j < height && image[((y + j) * width + x + maskX) * 3] == 255) {
                            foundNext = true;
                            x += maskX;
                            y += j;
                            len12++;
                            if (poly->data[((y + j) * width + x + maskX) * 3 + 2] == 255)
                                overlap12++;
                        }
                    }
                } else if (!dir && x + maskX < width) {
                    for (int j = maskY; j >= -maskY && !foundNext; --j) {
                        if (y + j >= 0 && y + j < height && image[((y + j) * width + x + maskX) * 3] == 255) {
                            foundNext = true;
                            x += maskX;
                            y += j;
                            len12++;
                            if (poly->data[((y + j) * width + x + maskX) * 3 + 2] == 255)
                                overlap12++;
                        }
                    }
                }
                maskX++;
                maskY++;
            }
        }


        // L23
        x = group[1].x, y = group[1].y;
        int len23 = 0, overlap23 = 0;
        dir = false;
        if (x < group[2].x)
            dir = true;
        while (y <= group[2].y) {
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
                            len23++;
                            if (poly->data[((y + maskY) * width + x + j) * 3 + 2] == 255)
                                overlap23++;
                        }
                    }
                } else if (!dir && y + maskY < height) {
                    for (int j = -maskX; j <= maskX && !foundNext; ++j) {
                        if (x + j >= 0 && x + j < width &&
                        image[((y + maskY) * width + x + j) * 3] == 255) {
                            foundNext = true;
                            x += j;
                            y += maskY;
                            len23++;
                            if (poly->data[((y + maskY) * width + x + j) * 3 + 2] == 255)
                                overlap23++;
                        }
                    }
                }
                maskX++;
                maskY++;
            }
        }

        // L34
        x = group[3].x, y = group[3].y;
        int len34 = 0, overlap34 = 0;
        dir = false;
        if (y < group[2].y)
            dir = true;
        while(x <= group[2].x) {
            bool foundNext = false;
            int maskX = 1, maskY = 1;
            while (!foundNext) {
                if (dir && x + maskX < width) {
                    for (int j = -maskY; j <= maskY && !foundNext; ++j) {
                        if (y + j >= 0 && y + j < height && image[((y + j) * width + x + maskX) * 3] == 255) {
                            foundNext = true;
                            x += maskX;
                            y += j;
                            len34++;
                            if (poly->data[((y + j) * width + x + maskX) * 3 + 2] == 255)
                                overlap34++;
                        }
                    }
                } else if (!dir && x + maskX < width) {
                    for (int j = maskY; j >= -maskY && !foundNext; --j) {
                        if (y + j >= 0 && y + j < height && image[((y + j) * width + x + maskX) * 3] == 255) {
                            foundNext = true;
                            x += maskX;
                            y += j;
                            len34++;
                            if (poly->data[((y + j) * width + x + maskX) * 3 + 2] == 255)
                                overlap34++;
                        }
                    }
                }
                maskX++;
                maskY++;
            }
        }


        // L41
        x = group[0].x, y = group[0].y;
        int len41 = 0, overlap41 = 0;
        dir = false;
        if (x < group[3].x)
            dir = true;
        while (y <= group[3].y) {
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
                            len41++;
                            if (poly->data[((y + maskY) * width + x + j) * 3 + 2] == 255)
                                overlap41++;
                        }
                    }
                } else if (!dir && y + maskY < height) {
                    for (int j = -maskX; j <= maskX && !foundNext; ++j) {
                        if (x + j >= 0 && x + j < width &&
                            image[((y + maskY) * width + x + j) * 3] == 255) {
                            foundNext = true;
                            x += j;
                            y += maskY;
                            len41++;
                            if (poly->data[((y + maskY) * width + x + j) * 3 + 2] == 255)
                                overlap41++;
                        }
                    }
                }
                maskX++;
                maskY++;
            }
        }

        //printf("%i %i %i %i %i %i %i %i\n", len12, overlap12, len23, overlap23, len34, overlap34, len41, overlap41);

        float fr12 = overlap12 * 1.0f / len12;
        float fr23 = overlap23 * 1.0f / len23;
        float fr34 = overlap34 * 1.0f / len34;
        float fr41 = overlap41 * 1.0f / len41;

        // First threshold
        if(fr12 >= fillRatioLOne && fr23 >= fillRatioLOne && fr34 >= fillRatioLOne && fr41 >= fillRatioLOne &&
           (fr12 + fr23 + fr34 + fr41 / 4) >= fillRatioHOne){

            // Check if this match group is inside another
            bool found = false;
            for (int y = 0; y < matchFillRatioOne.size() && !found; ++y) {
                vector<Point> checkGroup = matchFillRatioOne.at(y);
                if(checkGroup[0].x <= group[0].x && checkGroup[0].y <= group[0].y &&
                        checkGroup[1].x >= group[1].x && checkGroup[1].y <= group[1].y &&
                        checkGroup[2].x >= group[2].x && checkGroup[2].y >= group[2].y &&
                        checkGroup[3].x <= group[3].x && checkGroup[3].y >= group[3].y)
                    found = true;
            }
            if(!found)
                matchFillRatioOne.push_back(group);

            // Second threshold
            if(fr12 >= fillRatioLTwo && fr23 >= fillRatioLTwo && fr34 >= fillRatioLTwo && fr41 >= fillRatioLTwo &&
            (fr12 + fr23 + fr34 + fr41 / 4) >= fillRatioHTwo){
                // Check if this match group is inside another
                bool found = false;
                for (int y = 0; y < matchFillRatioTwo.size() && !found; ++y) {
                    vector<Point> checkGroup = matchFillRatioTwo.at(y);
                    if(checkGroup[0].x <= group[0].x && checkGroup[0].y <= group[0].y &&
                       checkGroup[1].x >= group[1].x && checkGroup[1].y <= group[1].y &&
                       checkGroup[2].x >= group[2].x && checkGroup[2].y >= group[2].y &&
                       checkGroup[3].x <= group[3].x && checkGroup[3].y >= group[3].y)
                        found = true;
                }
                if(!found)
                    matchFillRatioTwo.push_back(group);
            }
            //printf("trovato!\n");
        }
    }

    if(matchFillRatioOne.size() <= 1)
        matchFillRatio = matchFillRatioOne;
    else
        matchFillRatio = matchFillRatioTwo;

    return Utilities::seconds() - time;
}
