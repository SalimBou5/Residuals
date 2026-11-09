#include <string>
#include <iostream>
#include <fstream>
#include "include/MEP_class.h"
int main()
{
    const double speed_ms  = 1.2;
    const double speed_kmh = speed_ms * 3.6;   // model works in km/h (baseline = 3.0 in calcWeightings)
    const double elevation = 0.0;              // level walking

    MEP_class model;                           // reads dat/*.txt (run from the folder containing dat/)
    model.calcMEP(elevation, speed_kmh);       // fills model.weightings[4][15] and model.mep[200][15]

    const char* names[15] = {"TA","Sol","Per","VastLat","VastMed","RFem","Sar","Add",
                             "GlutMed","TFL","GastLat","GastMed","BFem","Semi","GlutMax"};
    std::ofstream f("mep_1p2ms.csv");
    f << "gait_cycle_pct";
    for (int m = 0; m < 15; m++) f << "," << names[m];
    f << "\n";
    for (int i = 0; i < 200; i++) {
        f << (100.0 * i / 199.0);              // 200 samples over the gait cycle
        for (int m = 0; m < 15; m++) f << "," << model.mep[i][m];
        f << "\n";
    }
    std::cout << "TA at 50% cycle: " << model.getMEP(100, "TA") << std::endl;
    return 0;
}
