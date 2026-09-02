#pragma once


#ifndef TRACK_H
#define TRACK_H

#include "LightTracker.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <chrono>

using namespace cv;
using namespace std;

class Track {
public:
	void tracker_init();
	std::pair<cv::Mat, int> tracker_run(cv::Mat image, cv::Rect box,
                                          int ptz_blind_phase = 0);
        void release();

private:
	LightTracker tracker;
	int frame_id;

};

#endif
