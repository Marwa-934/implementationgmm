// gmm.h
#pragma once

#include <opencv2/opencv.hpp>

struct Rank {
    float val;
    int index_unidim;
};

class GMM {
public:
    GMM(cv::Mat& premiere_trame, bool repeter_initialisation, int n);
    ~GMM();

    void initialisation(cv::Mat& currentFrame, int n);
    void traitement(const cv::Mat& currentFrame);
    void traitementPixel(int i, int j);
    bool determiner_foreground(int i, int j);

private:
    float* w;
    float* moyenne;
    float* deviation_std;
    float* difference;
    float update;
    cv::Mat frame;
    cv::Mat gray_frame;
    cv::Mat foreground;
    cv::Mat foreground_mask;
    cv::Mat background_pixel;

    const int composant_gaussien;
    const int composant_background;
    const float seuilDev;
    const float learning_rate;
    const float seuil;
    const float sd_initiale;
    const int largeur;
    const int hauteur;
};
