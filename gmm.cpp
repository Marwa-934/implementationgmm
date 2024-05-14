// gmm.cpp
#include <opencv2/opencv.hpp>
#include "gmm.h"

using namespace cv;
using namespace std;

// Forward declarations
static int ThreeDMapping(int largeur, int w, int hauteur, int h, int C, int c_index);
static bool comparer_func(const Rank& r1, const Rank& r2);

GMM::GMM(Mat& premiere_trame, bool repeter_initialisation, int n) :
    composant_gaussien(2), composant_background(2), seuilDev(2.5f), learning_rate(0.001f), seuil(0.4f), sd_initiale(6),
    largeur(premiere_trame.cols), hauteur(premiere_trame.rows)
{
    premiere_trame.copyTo(frame); // Copy the first frame to the current frame
    cvtColor(frame, gray_frame, COLOR_BGR2GRAY); // Convert the frame to grayscale

    // Allocate memory for mean, weights, standard deviation, and difference
    w = new float[largeur * hauteur * composant_gaussien];
    moyenne = new float[largeur * hauteur * composant_gaussien];
    deviation_std = new float[largeur * hauteur * composant_gaussien];
    difference = new float[largeur * hauteur * composant_gaussien];
    update = learning_rate / (1 / (float)composant_gaussien);

    // Random initialization of mean values
    for (int i = 0; i < largeur * hauteur * composant_gaussien; i++) {
        moyenne[i] = (float)(rand() % 256);
    }

    // Initialization of weights, standard deviation, and difference
    fill(w, w + largeur * hauteur * composant_gaussien, 1 / (float)composant_gaussien);
    fill(deviation_std, deviation_std + largeur * hauteur * composant_gaussien, 1 / (float)composant_gaussien);
    fill(difference, difference + largeur * hauteur * composant_gaussien, 1 / (float)composant_gaussien);

    // Repeat initialization for the first frame to generate a robust model
    if (repeter_initialisation) {
        initialisation(premiere_trame, n);
    }
}

GMM::~GMM()
{
    // Free allocated memory
    delete[] w;
    delete[] moyenne;
    delete[] deviation_std;
    delete[] difference;
}

void GMM::initialisation(Mat& currentFrame, int n)
{
    for (int i = 0; i <= n; i++) {
        traitement(currentFrame);
    }
}

void GMM::traitement(const Mat& currentFrame)
{
    assert(currentFrame.cols == frame.cols && currentFrame.rows == frame.rows);

    currentFrame.copyTo(frame);
    cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

    for (int i = 0; i < hauteur; i++) {
        for (int j = 0; j < largeur; j++) {
            for (int q = 0; q < composant_gaussien; q++) {
                int index_unidim = ThreeDMapping(largeur, j, hauteur, i, composant_gaussien, q);
                difference[index_unidim] = fabsf((float)gray_frame.at<uchar>(i, j) - moyenne[index_unidim]);
            }
        }
    }

    this->foreground.release();
    this->foreground.create(largeur, hauteur, CV_8UC1); // Changed DataType<int> to CV_8UC1
    this->foreground_mask.release();
    this->foreground_mask.create(largeur, hauteur, CV_8UC1); // Changed size to match frame
    this->background_pixel.release();
    this->background_pixel.create(largeur, hauteur, CV_32FC1);

    for (int i = 0; i < hauteur; i++) {
        for (int j = 0; j < largeur; j++) {
            traitementPixel(i, j); // Call to traitementPixel
        }
    }
}

void GMM::traitementPixel(int i, int j)
{
    bool match = false;
    float somme_composant = 0;
    int index = ThreeDMapping(largeur, j, hauteur, i, composant_background, 0);
    int pix_val = gray_frame.at<uchar>(i, j);

    for (int k = 0; k < composant_gaussien; ++k) {
        int index_k = index + k;
        if (fabsf(difference[index_k]) <= seuilDev * difference[index_k]) {
            match = true;
            w[index_k] = (1 - learning_rate) * w[index_k] + learning_rate;
            update = learning_rate / w[index_k];
            moyenne[index_k] = (1 - update) * moyenne[index_k] + update * pix_val;
            deviation_std[index_k] = sqrtf((1 - update) * (deviation_std[index_k] * deviation_std[index_k]) +
                update * ((pix_val - moyenne[index_k]) * (pix_val - moyenne[index_k])));
            somme_composant += w[index_k];
        }
        else {
            w[index_k] = (1 - learning_rate) * w[index_k];
            somme_composant += w[index_k];
        }
    }

    int min_index = -1;
    float min_weight = FLT_MAX;
    for (int k = 0; k < composant_gaussien; ++k) {
        int index_k = index + k;
        w[index_k] /= somme_composant;
        if (!match) {
            if (w[index_k] < min_weight) {
                min_index = index_k;
                min_weight = w[index_k];
            }
        }
    }

    background_pixel.at<float>(i, j) = 0;
    for (int k = 0; k < composant_gaussien; ++k) {
        int index_k = index + k;
        background_pixel.at<float>(i, j) += moyenne[index_k] * w[index_k];
    }

    if (!match) {
        assert(min_index != -1);
        moyenne[min_index] = static_cast<float>(pix_val);
        deviation_std[min_index] = sd_initiale;
    }

    Rank* rank_array = new Rank[composant_gaussien];
    for (int k = 0; k < composant_gaussien; ++k) {
        int index_k = index + k;
        rank_array[k].val = w[index_k] / deviation_std[index_k];
        rank_array[k].index_unidim = k;
    }

    // Sort rank_array based on val
    sort(rank_array, rank_array + composant_gaussien, comparer_func);

    // Reset match flag
    match = false;

    // Loop through rank_array with appropriate termination condition
    int k = 0;
    while (!match && k < composant_gaussien) {
        int index_k = index + rank_array[k].index_unidim;
        if (w[index_k] >= seuil) {
            if (fabs(difference[index_k]) <= seuilDev * deviation_std[index_k]) {
                foreground.at<uchar>(i, j) = 0;
                match = true;
            }
            else {
                foreground.at<uchar>(i, j) = pix_val;
            }
        }
        ++k;
    }

    // Clean up dynamically allocated memory
    delete[] rank_array;

    // Set foreground mask pixel value
    foreground_mask.at<uchar>(i, j) = match ? 255 : 0;
}

bool GMM::determiner_foreground(int i, int j)
{
    return foreground_mask.at<uchar>(i, j) == 255; // Check if foreground mask pixel is white
}

int ThreeDMapping(int largeur, int w, int hauteur, int h, int C, int c_index)
{
    return (h * largeur + w) * C + c_index;
}

bool comparer_func(const Rank& r1, const Rank& r2)
{
    return r1.val > r2.val;
}
