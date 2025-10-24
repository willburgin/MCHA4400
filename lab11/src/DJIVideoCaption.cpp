#include <cassert>
#include <cstdio>
#include <string>
#include <filesystem>
#include "DJIVideoCaption.h"

std::vector<DJIVideoCaption> getVideoCaptions(const std::filesystem::path & captionPath)
{
    assert(std::filesystem::exists(captionPath));
    std::vector<DJIVideoCaption> caps;
    
    FILE* file = fopen(captionPath.string().c_str(), "r");
    if (!file) {
        return caps;
    }
    
    int frameNum, year, month, day, hour, min, sec, msec, usec;
    int timeHour, timeMin, timeSec, timeMsec;  // From SRT timestamp
    int iso, fnum;
    double shutterSpeed, latitude, longitude, altitude;
    
    while (fscanf(file,
        "%*d " // subtitle number
        "%d:%d:%d,%d --> %*d:%*d:%*d,%*d " // timestamp range (use start time)
        "<font size=\"36\">FrameCnt : %d, DiffTime : %*dms "
        "%d-%d-%d %d:%d:%d,%d,%d "
        "[iso : %d] [shutter : 1/%lf] [fnum : %d] "
        "[ev : %*d] [ct : %*d] [color_md : %*[^]]] [focal_len : %*d] "
        "[latitude : %lf] [longtitude : %lf] [altitude: %lf] </font> ",
        &timeHour, &timeMin, &timeSec, &timeMsec, &frameNum,
        &year, &month, &day, &hour, &min, &sec, &msec, &usec,
        &iso, &shutterSpeed, &fnum, &latitude, &longitude, &altitude) == 19)
    {
        DJIVideoCaption cap;
        cap.frameNum = frameNum;
        
        // Use SRT timestamp for video time
        cap.time = timeHour * 3600.0 + timeMin * 60.0 + timeSec + timeMsec / 1000.0;
        cap.iso = iso;
        cap.shutterHz = shutterSpeed;
        cap.fnum = fnum / 100.0;
        cap.latitude = latitude;
        cap.longitude = longitude;
        cap.altitude = altitude;
        
        caps.push_back(cap);
    }
    
    fclose(file);
    return caps;
}
