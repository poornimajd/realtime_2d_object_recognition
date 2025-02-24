/*
  Poornima Jaykumar Dharamdasani
  feb 2025

  Showing different effects on video based on the key press.

*/
#include <opencv2/opencv.hpp>   
#include <opencv2/imgproc.hpp> 
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>


#include "threshold.h"
#include "morph.h"
#include "segment.h"
#include <filesystem>
#include <fstream>



struct RegionFeatures_loaded {

    std::string label;    
    int region_id = -1;  // -1 means "no valid region"
    cv::RotatedRect bounding_box = cv::RotatedRect(cv::Point2f(-1, -1), cv::Size2f(0, 0), 0);
    double aspectRatio = -1.0;  // -1 to indicate undefined value
    double percentfilled = -1.0;  // -1 means invalid calculation
    cv::Vec2d major_axis = cv::Vec2d(0, 0);  // Default to zero vector
    cv::Vec2d minor_axis = cv::Vec2d(0, 0);
    cv::Point2f centroid = cv::Point2f(-1, -1);  
    double huMoments[7];


    float stdev_aspectRatio=1.0;
    float stdev_percentfilled =1.0;
    float stdev_huMoments[7] = {1, 1, 1, 1, 1, 1, 1};

};

void display_var(cv::Mat &img, cv::Mat &thresholded_img);
// void findfeatures(cv::Mat &src, cv::Mat &dst);
namespace fs = std::filesystem;

/*
 Loads a CSV file and extracts region features.
 Arguments:
 - csvFilePath: Path to the CSV file.

 Functionality:
 - Opens the CSV file.
 - Reads each line and extracts region properties.
 - Parses values into a RegionFeatures_loaded structure.
 - Stores the extracted regions in a vector.
 Returns:
 - A vector of RegionFeatures_loaded objects.
 */

std::vector<RegionFeatures_loaded> loadCSVAndExtractMaxRegions(std::string &csvFilePath)
{
        std::ifstream file(csvFilePath);
        if (!file.is_open()) {
        std::cerr << "Error opening CSV file: " << csvFilePath << std::endl;
        return {};
        }

        std::string line;
 
        // std::map<std::string, RegionFeatures_loaded> maxRegions; 
        std::vector<RegionFeatures_loaded> loaded_features;

        while(std::getline(file,line))
        {

        std::stringstream ss(line);
        std::string label;
        RegionFeatures_loaded region;
        std::string token;

        std::getline(ss, region.label, ',');  

        std::getline(ss, token, ','); region.region_id = std::stoi(token);
        std::getline(ss, token, ','); region.aspectRatio = std::stof(token);
        std::getline(ss, token, ','); region.percentfilled = std::stof(token);
        std::getline(ss, token, ','); region.centroid.x = std::stof(token);
        std::getline(ss, token, ','); region.centroid.y = std::stof(token);
        std::getline(ss, token, ','); region.bounding_box.center.x = std::stof(token);
        std::getline(ss, token, ','); region.bounding_box.center.y = std::stof(token);
        std::getline(ss, token, ','); region.bounding_box.size.width = std::stof(token);
        std::getline(ss, token, ','); region.bounding_box.size.height = std::stof(token);
        std::getline(ss, token, ','); region.bounding_box.angle = std::stof(token);
        std::getline(ss, token, ','); region.major_axis[0] = std::stof(token);
        std::getline(ss, token, ','); region.major_axis[1] = std::stof(token);
        std::getline(ss, token, ','); region.minor_axis[0] = std::stof(token);
        std::getline(ss, token, ','); region.minor_axis[1] = std::stof(token);

        for (int i = 0; i < 7; i++) {
        std::getline(ss, token, ',');
        region.huMoments[i] = std::stof(token);
    }

        loaded_features.push_back(region);
        
        }
        file.close();

        
        return loaded_features;


}

/*
 Computes the standard deviation of a vector of values.
 Arguments:
 - values: Vector of float numbers.

 Functionality:
 - Computes the mean of the values.
 - Calculates variance using the sample standard deviation formula (n-1).
 - Returns the square root of the variance.
 Returns:
 - Standard deviation as a float.
 */

float computeStdDev(std::vector<float> &values) {
    if (values.size() <= 1) return 0.0; // Avoid division by zero

    float sum = 0, mean, variance = 0;
    for (float val : values) sum += val;
    mean = sum / values.size();

    for (float val : values) variance += (val - mean) * (val - mean);
    variance /= (values.size() - 1); // Sample standard deviation (n-1)
    // std::cout << "what is the variance??: " << variance << std::endl;
    // std::cout << "what is the size??: " << values.size() << std::endl;
    return std::sqrt(variance);
}

/*
 Computes the standard deviation for region features.
 Arguments:
 - maxRegions: Vector of RegionFeatures_loaded objects.

 Functionality:
 - Collects aspect ratios, percent filled, and Hu moments from all regions.
 - Computes standard deviation for each feature.
 - Assigns the computed values to each region.
 - If a standard deviation is too small (< 0.001), assigns a default value of 1.0.
 - Prints the computed standard deviations.
 */

void computeStdDevForRegions(std::vector<RegionFeatures_loaded>& maxRegions) {
    std::vector<float> aspectRatios, percentFilleds;
    std::vector<std::vector<float>> huMomentValues(7);

    // Collect feature values
    for (const auto& region : maxRegions) {
        aspectRatios.push_back(region.aspectRatio);
        percentFilleds.push_back(region.percentfilled);
        for (int i = 0; i < 7; i++) {
            huMomentValues[i].push_back(region.huMoments[i]);
        }
       
    }

    // Compute standard deviation for each feature
    float stdev_aspectRatio = computeStdDev(aspectRatios);
    float stdev_percentfilled = computeStdDev(percentFilleds);

    float stdev_huMoments[7];
    for (int i = 0; i < 7; i++) {
        stdev_huMoments[i] = computeStdDev(huMomentValues[i]);
    }

    // Assign computed standard deviations to each region
    for (auto& region : maxRegions) {
        region.stdev_aspectRatio = (stdev_aspectRatio<0.001)?1.0f:stdev_aspectRatio;

        region.stdev_percentfilled = (stdev_percentfilled < 0.001) ? 1.0f : stdev_percentfilled;

        for (int i = 0; i < 7; i++) {
            region.stdev_huMoments[i] = (stdev_huMoments[i] < 0.001) ? 1.0f : stdev_huMoments[i];
        }


        std::cout << "Standard Deviations for Region: " << region.label << std::endl;
        std::cout << "Aspect Ratio std: " << region.stdev_aspectRatio << std::endl;
        std::cout << "Percent Filled: " << region.stdev_percentfilled << std::endl;

        std::cout << "Hu Moments Standard Deviations: ";
        for (int i = 0; i < 7; i++) {
            std::cout << region.stdev_huMoments[i] << " ";
        }
        std::cout << std::endl;
    }

}

// float safe_div(float numerator, float denominator) {
//     return (std::abs(denominator) > 1e-6) ? (numerator / denominator) : (numerator / 1e-6);
// }

/*
 Computes the label for a region using Euclidean distance.
 Arguments:
 - region_features: Features of the region to classify.
 - maxRegions: Vector of reference regions with known labels.

 Functionality:
 - Calculates the normalized squared difference for aspect ratio, percent filled, and Hu moments.
 - Computes the Euclidean distance between the input region and each reference region.
 - Finds the closest match with the minimum distance.
 - Assigns the label of the closest reference region.
 Returns:
 - The predicted label as a string.
 */

std::string compute_the_label_euclidean(RegionFeatures &region_features, std::vector<RegionFeatures_loaded> &maxRegions)
{       std::string predicted_label;
        float min_distance = std::numeric_limits<float>::max();
        float huMomentsdiff[7];

        for(int j=0;j<maxRegions.size();j++)
        {
        
            float aspectdiff = std::pow((region_features.aspectRatio - maxRegions[j].aspectRatio) / maxRegions[j].stdev_aspectRatio, 2);

            float percentdiff = std::pow((region_features.percentfilled - maxRegions[j].percentfilled) / maxRegions[j].stdev_percentfilled, 2);

            float huSum = 0;  // Sum of Hu Moments squared differences
                for (int i = 0; i < 7; i++)
                {
                    huMomentsdiff[i] = std::pow((region_features.huMoments[i] - maxRegions[j].huMoments[i]) / maxRegions[j].stdev_huMoments[i], 2);
                    huSum += huMomentsdiff[i];  // Accumulate squared differences
                }


        float total_distance = std::sqrt(
                aspectdiff + percentdiff + huSum);
        // std::cout<<total_distance<<std::endl;

         if (total_distance < min_distance) {
                min_distance = total_distance;
                predicted_label = maxRegions[j].label;
            }

        //take sqrt of total dis and store it, see which is minimum and give that label. to this region. and return the labels for that image.

        }

        return predicted_label;


        //}
        
}
//Extension
/*
 Computes the label for a region using cosine similarity.
 Arguments:
 - region_features: Features of the region to classify.
 - maxRegions: Vector of reference regions with known labels.

 Functionality:
 - Computes cosine similarity using the dot product and magnitudes of aspect ratio, percent filled, and Hu moments.
 - Converts similarity to cosine distance (1 - similarity).
 - Finds the closest match with the smallest cosine distance.
 - Assigns the label of the closest reference region.
 Returns:
 - The predicted label as a string.
 */

std::string compute_the_label_cosine(RegionFeatures &region_features, std::vector<RegionFeatures_loaded> &maxRegions) {
    std::string predicted_label;
    float min_distance = std::numeric_limits<float>::max();

    for (int j = 0; j < maxRegions.size(); j++) {
        // Compute dot product and magnitudes
        float dot_product = (region_features.aspectRatio * maxRegions[j].aspectRatio) +
                            (region_features.percentfilled * maxRegions[j].percentfilled);
        float magA = std::sqrt(std::pow(region_features.aspectRatio, 2) + std::pow(region_features.percentfilled, 2));
        float magB = std::sqrt(std::pow(maxRegions[j].aspectRatio, 2) + std::pow(maxRegions[j].percentfilled, 2));

        float hu_dot_product = 0, magA_hu = 0, magB_hu = 0;
        for (int i = 0; i < 7; i++) {
            hu_dot_product += (region_features.huMoments[i] * maxRegions[j].huMoments[i]);
            magA_hu += std::pow(region_features.huMoments[i], 2);
            magB_hu += std::pow(maxRegions[j].huMoments[i], 2);
        }

        float cosine_similarity = (dot_product + hu_dot_product) / ((magA * magB) + std::sqrt(magA_hu) * std::sqrt(magB_hu));
        float cosine_distance = 1 - cosine_similarity; // Convert to distance

        if (cosine_distance < min_distance) {
            min_distance = cosine_distance;
            predicted_label = maxRegions[j].label;
        }
    }

    return predicted_label;
}

//Extension
/*
 Computes the label for a region using scaled L1 (Manhattan) distance.
 Arguments:
 - region_features: Features of the region to classify.
 - maxRegions: Vector of reference regions with known labels.

 Functionality:
 - Computes the absolute difference for aspect ratio, percent filled, and Hu moments.
 - Normalizes each difference using the standard deviation of the reference region.
 - Computes the total distance by summing up the normalized differences.
 - Finds the closest match with the smallest total distance.
 - Assigns the label of the closest reference region.
 Returns:
 - The predicted label as a string.
 */

std::string compute_the_label_scaled_L1(RegionFeatures &region_features, std::vector<RegionFeatures_loaded> &maxRegions) {
    std::string predicted_label;
    float min_distance = std::numeric_limits<float>::max();

    for (int j = 0; j < maxRegions.size(); j++) {
        float aspectdiff = std::abs(region_features.aspectRatio - maxRegions[j].aspectRatio) / maxRegions[j].stdev_aspectRatio;
        float percentdiff = std::abs(region_features.percentfilled - maxRegions[j].percentfilled) / maxRegions[j].stdev_percentfilled;

        float huSum = 0;
        for (int i = 0; i < 7; i++) {
            huSum += std::abs(region_features.huMoments[i] - maxRegions[j].huMoments[i]) / maxRegions[j].stdev_huMoments[i];
        }

        float total_distance = aspectdiff + percentdiff + huSum;

        if (total_distance < min_distance) {
            min_distance = total_distance;
            predicted_label = maxRegions[j].label;
        }
    }

    return predicted_label;
}

/*
 Classifies an object using k-Nearest Neighbors (k-NN).
 Arguments:
 - region_features: Features of the region to classify.
 - maxRegions: Vector of reference regions with known labels.
 - k: Number of nearest neighbors to consider.

 Functionality:
 - Computes the Euclidean distance for aspect ratio, percent filled, and Hu moments.
 - Stores distances along with labels.
 - Sorts the distances in ascending order.
 - Selects the label with the smallest distance.
 Returns:
 - The predicted label as a string.
 */

std::string classifyObjectKNN(RegionFeatures &region_features, std::vector<RegionFeatures_loaded> &maxRegions, int k) {
    std::vector<std::pair<float, std::string>> distances;
     float huMomentsdiff[7];

    for(int j=0;j<maxRegions.size();j++)
    {
        
            float aspectdiff = std::pow((region_features.aspectRatio - maxRegions[j].aspectRatio) / maxRegions[j].stdev_aspectRatio, 2);

            float percentdiff = std::pow((region_features.percentfilled - maxRegions[j].percentfilled) / maxRegions[j].stdev_percentfilled, 2);

            float huSum = 0;  // Sum of Hu Moments squared differences
                for (int i = 0; i < 7; i++)
                {
                    huMomentsdiff[i] = std::pow((region_features.huMoments[i] - maxRegions[j].huMoments[i]) / maxRegions[j].stdev_huMoments[i], 2);
                    huSum += huMomentsdiff[i];  // Accumulate squared differences
                }


        float total_distance = std::sqrt(
                aspectdiff + percentdiff + huSum);
        distances.push_back({ total_distance, maxRegions[j].label});
    }

    // Sort distances
    std::sort(distances.begin(), distances.end());


    std::string bestLabel = distances[0].second;
    double minDistance = distances[0].first;

    for (int i = 1; i < k && i < distances.size(); ++i)
    {
        if (distances[i].first < minDistance)
        {
            minDistance = distances[i].first;
            bestLabel = distances[i].second;
        }
    }
    return bestLabel;

    // std::map<std::string, float> distanceSums;
    // for (int i = 0; i < k && i < distances.size(); i++) {
    //     distanceSums[distances[i].second] += distances[i].first;
    // }

    // // Find the label with the minimum sum of distances
    // std::string bestMatchLabel;
    // float minSumDistance = std::numeric_limits<float>::max();
    // for (const auto& pair : distanceSums) {
    //     if (pair.second < minSumDistance) {
    //         minSumDistance = pair.second;
    //         bestMatchLabel = pair.first;
    //     }
    // }

    // return bestMatchLabel;
}


int main(int argc, char *argv[]) {

        //take an input key, now if it c, ask to input even database file.
        //else if N, then as to inputfolder name,where the train images will be stored with one single csv file., whenever the user presses a 's'

        /*
         Handles user input for CSV file processing or creating a new database folder.
         
         Functionality:
         - Asks the user to input a CSV file ('c') or create a new folder ('N').
         - If 'c' is chosen, loads data from the given CSV file and computes standard deviations.
         - If 'N' is chosen, creates a new folder and an empty CSV file inside it.
         - If input is invalid, prints an error message and exits.

         Initializes video capture:
         - Opens a video file for processing.
         - Checks if the video device opens successfully.
         - Retrieves and prints the video frame size.
         - Creates two windows for displaying original and processed videos.

         Variables:
         - `frame`: Stores frames from the video.
         - `thresholded_frame`: Stores processed frames after applying thresholding.
         */

        char choice;
       std::cout << "Press 'c' to input a CSV file, or 'N' to create a new folder and CSV file (give full path for csv!): ";
       std::cin >> choice;
       std::string csvFilePath;
        std::string folderName;
        std::vector<RegionFeatures_loaded> maxRegions;
        if(choice=='c'||choice=='C')
        {
                std::cout << "Enter the path of the .csv file: ";
                std::cin >> csvFilePath;
                std::cout << "Using CSV file: " << csvFilePath << std::endl;
                maxRegions = loadCSVAndExtractMaxRegions(csvFilePath);
                computeStdDevForRegions(maxRegions);

        }
        else if(choice == 'N' || choice == 'n')
        {

        std::cout << "Enter the folder name to store the database: ";
        std::cin >> folderName;

        // Create the folder if it doesn't exist
        if (!fs::exists(folderName)) {
            fs::create_directory(folderName);
            std::cout << "Folder created: " << folderName << std::endl;
        } else {
            std::cout << "Folder already exists: " << folderName << std::endl;
        }


        // Create a CSV file inside the folder
        csvFilePath = folderName + "/" + folderName + ".csv";
        std::ofstream csvFile(csvFilePath);
        if (csvFile.is_open()) {
            std::cout << "CSV file created: " << csvFilePath << std::endl;
            csvFile.close();
        } else {
            std::cerr << "Failed to create CSV file: " << csvFilePath << std::endl;
            return -1;
        }

        }

        else {
        std::cerr << "Invalid input. Exiting..." << std::endl;
        return -1;
        }

        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture("/root/sys_path/assign3/output_test.avi");
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Original Video", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Modified Video", cv::WINDOW_AUTOSIZE);


        cv::Mat frame, thresholded_frame; // initializing different frames, needed for processing later.

        /*
         Processes video frames in a loop.

         Functionality:
         - Captures frames from the video stream.
         - Applies dynamic thresholding to segment objects.
         - Performs morphological operations to refine segmentation.
         - Extracts features from segmented regions.
         - Prints extracted features like aspect ratio, centroid, and bounding box.

         If a CSV file is loaded:
         - Compares the detected regions with reference regions from the CSV.
         - Uses Euclidean distance to classify the region.
         - Displays the predicted label on the segmented image.
         - If no label is found, assigns "UNKNOWN".

         Displays the labeled segmented image.
         */

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }
               thresholded_frame = dynamic_threshold(frame);
               // display_var(frame, thresholded_frame);

                
                cv::Mat morphed;
                morphological_op(thresholded_frame, morphed);
                //cv::Mat featured_image;
                cv::Mat segmented_img;
                cv::Mat save_seg;
                std::vector<RegionFeatures> all_features = segment_and_features(morphed,segmented_img);
                save_seg =segmented_img.clone();

                for (auto& region : all_features) {
                    std::cout << "Region ID: " << region.region_id << "\n";
                    std::cout << "Aspect Ratio: " << region.aspectRatio << "\n";
                    std::cout << "Percent Filled: " << region.percentfilled << "%\n";
                    std::cout << "Centroid: (" << region.centroid.x << ", " << region.centroid.y << ")\n";

                    std::cout << "Bounding Box Center: (" << region.bounding_box.center.x 
                              << ", " << region.bounding_box.center.y << ")\n";
                    std::cout << "Bounding Box Size: (" << region.bounding_box.size.width
                              << " x " << region.bounding_box.size.height << ")\n";
                    std::cout << "Bounding Box Angle: " << region.bounding_box.angle << "°\n";

                    
                    std::cout << "--------------------------------------\n";

                if(choice=='c' || choice=='C')
                {
                        //compare the loaded features with this images; ffeatures, get the albel, and display with label.
                        // get the label as output, input to function is std::vector<RegionFeatures> all_features  and comparing with std::vector<RegionFeatures_loaded> maxRegions
                        // display the image with the out put then
                        std::string label_to_display = compute_the_label_euclidean(region, maxRegions);
                        // std::string label_to_display = classifyObjectKNN(region, maxRegions,3);
                        if(label_to_display.size()==0)
                                label_to_display="UNKNOWN";

                        cv::putText(segmented_img, 
                                    label_to_display, 
                                    cv::Point(region.bounding_box.center.x, region.bounding_box.center.y), 
                                    cv::FONT_HERSHEY_SIMPLEX,  
                                    1.1,  
                                    cv::Scalar(0, 255, 0),  
                                    2,  
                                    cv::LINE_AA);  
     
                }
                
                //display
                /*
                 Visualizes detected regions and handles user interactions.

                 Functionality:
                 - Draws a red dot at the centroid of each detected region.
                 - Draws major and minor axes as lines in different colors.
                 - Draws the bounding box around the region.
                 - Displays the segmented image alongside the original frame.

                 User interactions:
                 - Press 'q' to quit the loop.
                 - Press 's' to save images and region features.

                 When 's' is pressed:
                 - Prompts the user to enter a label for the image.
                 - Saves multiple versions of the image (original, thresholded, morphed, segmented).
                 - Appends region features and label to a CSV file.
                 */
                
                cv::circle(segmented_img, cv::Point(region.centroid.x, region.centroid.y), 5, cv::Scalar(0, 0, 255), -1);


                
                cv::line(segmented_img, cv::Point(region.centroid.x, region.centroid.y),cv::Point(region.centroid.x+static_cast<int>(100*region.major_axis[0]),region.centroid.y+static_cast<int>(100*region.major_axis[1])),CV_RGB(0, 255, 255), 2);
                cv::line(segmented_img, cv::Point(region.centroid.x, region.centroid.y),cv::Point(region.centroid.x+static_cast<int>(100*region.minor_axis[0]),region.centroid.y+static_cast<int>(100*region.minor_axis[1])),CV_RGB(255, 0, 255), 2);


                cv::Point2f boxpoints[4];
                region.bounding_box.points(boxpoints);

                for (int i = 0; i < 4; i++)
                    {
                        cv::line(segmented_img, boxpoints[i], boxpoints[(i + 1) % 4], CV_RGB(255, 255, 255), 2);
                    }

                }
                // cv::Mat contor_img;
                // findfeatures(segmented, contor_img);

                
                display_var(frame,segmented_img);
                char key = cv::waitKey(1000/0.5);

                if (key == 'q') { 
                    break;
                }

                else if(key=='s')
                {
                    std::string label;
                    std::cout << "Enter label for the image: ";
                    std::cin >> label;
                    if(folderName.size()==0)
                        folderName = "savetest";
                    std::string imgPath = "/root/sys_path/assign3/"+folderName + "/" + label + "op.png";
                    //std::cout<<imgPath<<std::endl;
                    std::string imgPath1 = "/root/sys_path/assign3/"+folderName + "/" + label + "t.png";
                    std::string imgPath2 = "/root/sys_path/assign3/"+folderName + "/" + label + "m.png";
                    std::string imgPath3 = "/root/sys_path/assign3/"+folderName + "/" + label + "p.png";
                    std::string imgPath4 = "/root/sys_path/assign3/"+folderName + "/" + label + "c.png";
                    cv::imwrite(imgPath, frame);
                    cv::imwrite(imgPath1, thresholded_frame);
                    cv::imwrite(imgPath2, morphed);
                    cv::imwrite(imgPath3, segmented_img);
                    cv::imwrite(imgPath4, save_seg);
                    // cv::imwrite(imgPath, segmented_img);
                    std::cout << "Image saved: " << imgPath << std::endl;

                    std::ofstream csvFile("/root/sys_path/assign3/"+csvFilePath, std::ios::app);
                    std::cout<<"/root/sys_path/assign3/"+csvFilePath<<std::endl;

                    if (!csvFile.is_open()) {
                        std::cerr << "Error opening CSV file for writing.\n";
                        continue;
                    }
                    
                    for (const auto& region : all_features) {
                        csvFile << label << ","
                                << region.region_id << ","
                                << region.aspectRatio << ","
                                << region.percentfilled << ","
                                << region.centroid.x << "," << region.centroid.y << ","
                                << region.bounding_box.center.x << "," << region.bounding_box.center.y << ","
                                << region.bounding_box.size.width << "," << region.bounding_box.size.height << ","
                                << region.bounding_box.angle << ","
                                << region.major_axis[0] << "," << region.major_axis[1] << ","
                                << region.minor_axis[0] << "," << region.minor_axis[1];

                                for (int i = 0; i < 7; i++) {
                                        csvFile << "," << region.huMoments[i];  
                                    }
                                csvFile << "\n";
                    }


                    csvFile.close();
                    std::cout << "Label and features saved in CSV file.\n";

                }
        }

        delete capdev;
        return(0);
}
/*
 Displays the provided image in a window.
 Arguments:
 - img: Reference to the image matrix to be displayed.
 */

void display_var(cv::Mat &img, cv::Mat &processed_img)
{

        cv::imshow("Original Video",img);
        cv::imshow("Modified Video",processed_img);


}




