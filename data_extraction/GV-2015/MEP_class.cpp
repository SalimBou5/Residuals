#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <locale.h>
#include "include/MEP_class.h"

using namespace std;

MEP_class::MEP_class()
{
	setlocale(LC_NUMERIC, "C");
	MEP_class::importRegression_coef();
	MEP_class::importXP();
	MEP_class::importBaseline();
		
}
	
void MEP_class::importXP()
{
    try
    {
        ifstream data("dat/gaussXP.txt");
        //data.exceptions(ifstream::failbit | ifstream::badbit);
        int i = 0;
        int j = 0;
        string line;
        string cell;
        double cell_double = 0;
    
        cout << "importing XP" << endl;
        while (getline(data, line))
        {   
			
            stringstream lineStream(line);
            string cell;
            while (getline(lineStream,cell,','))
            {
				
                cell_double = atof(cell.c_str());
                MEP_class::gaussXP[j][i] = cell_double;
                j++;
                //cout << cell.c_str() << "--" << cell_double << endl;
            }
            i++;
            j = 0;
        }
        data.close();
    }
    catch(ifstream::failure e)
    {
        cout<< e.what() << endl;
    }
}

void MEP_class::importRegression_coef()
{
    try{
        ifstream data("dat/regression.txt");
        //data.exceptions(ifstream::failbit | ifstream::badbit);
        int i = 0;
        int j = 0;
        string line;
        string cell;
        double cell_double = 0;
 
        cout << "importing regression " << endl;
        while (getline(data, line))
        {
            stringstream lineStream(line);
            string cell;
            while (getline(lineStream,cell,','))
            {
				
                cell_double = atof(cell.c_str());
                MEP_class::regression_coef[i][j] = cell_double;
                j++;
               // cout << "reg entered" << endl;
            }
            i++;
            j = 0;
        }
        data.close();
    }
    catch(ifstream::failure e)
    {
        cout<< e.what() << endl;
    }
}

void MEP_class::importBaseline()
{
    try
    {
        ifstream data("dat/baseline.txt");
        //data.exceptions(ifstream::failbit | ifstream::badbit);
        int i = 0;
        int j = 0;
        string line;
        string cell;
        double cell_double = 0;
    
        cout << "importing baseline" << endl;
        while (getline(data, line))
        {
            stringstream lineStream(line);
            
            string cell;
            while (getline(lineStream,cell,','))
            {
                cell_double = atof(cell.c_str());
                MEP_class::weightings_baseline[j][i] = cell_double;
                //cout << j << " - " << cell_double << endl;
                j++;
                //cout << j << " - " << i << endl;
            }
            i++;
            j = 0;
        }
        data.close();
    }
    catch(ifstream::failure e)
    {
        cout<< e.what() << endl;
    }
}

void MEP_class::calcWeightings(double elevation, double speed)
{
	double elevation_temp = 0.0;
	double speed_temp = 0.0;
	double correction = 0.0 ; 
	double constant = 0.0;	
	
	for (int comp=1;comp<=4;comp++)
	{	
		for (int musc=1;musc<=15;musc++)
		{			
			/* get the correction value at baseline: elevation =0, speed = 3*/
			//cout << "-------------------" << musc << comp << endl;
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("elevation", comp, 1)];
			elevation_temp = constant*(0.0*0.0);
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("elevation", comp, 2)];
			elevation_temp += (constant)*0.0;
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("elevation", comp, 3)];
			elevation_temp += constant;
			
			
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("speed", comp, 1)];
			speed_temp = constant*(3.0*3.0);
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("speed", comp, 2)];
			speed_temp += (constant)*3.0;
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("speed", comp, 3)];
			speed_temp += constant;
			
            
			correction = elevation_temp + speed_temp; 
			
			/* get the weighting value */		
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("elevation", comp, 1)];
			elevation_temp = constant*(elevation*elevation);
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("elevation", comp, 2)];
			elevation_temp += (constant)*elevation;
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("elevation", comp, 3)];
			elevation_temp += constant;
			
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("speed", comp, 1)];
			speed_temp = constant*(speed*speed);
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("speed", comp, 2)];
			speed_temp += (constant)*speed;
			constant = MEP_class::regression_coef[musc-1][MEP_class::w_pointer("speed", comp, 3)];
			speed_temp += constant; 
					
			MEP_class::weightings[comp-1][musc-1] = 
				MEP_class::weightings_baseline[comp-1][musc-1] + 
				elevation_temp + speed_temp - correction;	
			
			/* Deals with negative values */		
			if (MEP_class::weightings[comp-1][musc-1] < 0)
				MEP_class::weightings[comp-1][musc-1] = 0; 
				
			//cout << " speed: " << speed_temp << " elevation: " << elevation_temp << endl;
			//cout << "baseline " << MEP_class::weightings_baseline[comp-1][musc-1] << endl;	
			//cout << "weighting: " << MEP_class::weightings[comp-1][musc-1]<< endl;
			
			//cin >> key;
		}
	}
}

int MEP_class::w_pointer(string cond, int comp, int val)
{
	if (cond == "elevation")
	{
		return (comp-1)*3 + val-1;
	}
	
	if (cond == "speed")
	{
		return (comp-1)*3 + 12 + val-1;
	}
	
	return -1;
}

void MEP_class::calcMEP(double elevation, double speed)
{
	MEP_class::calcWeightings(elevation, speed);
	for (int i=0;i<200; i++)
	{
		for (int musc=0;musc<15; musc++)
		{
			MEP_class::mep[i][musc] = 0; 
			for (int comp=0;comp<4;comp++)
			{
				MEP_class::mep[i][musc] += MEP_class::gaussXP[i][comp]*MEP_class::weightings[comp][musc];
			}
		}
	}		
}

int indexMuscle(string muscle)
{
	if (muscle == "TA") 
		return 1;
	else
		return 0;	
}

int MEP_class::getMuscle_Index(string muscle)
{
    if (muscle == "TA")
    {
        return 0;
    } else if (muscle == "Sol")
    {
        return 1;
    } else if (muscle == "Per")
    {
        return 2;
    } else if (muscle == "VastLat")
    {
        return 3;
    } else if (muscle == "VastMed")
    {
        return 4;
    } else if (muscle == "RFem")
    {
        return 5;
    } else if (muscle == "Sar")
    {
        return 6;
    } else if (muscle == "Add")
    {
        return 7;
    } else if (muscle == "GlutMed")
    {
        return 8;
    } else if (muscle == "TFL")
    {
        return 9;
    } else if (muscle == "GastLat")
    {
        return 10;
    } else if (muscle == "GastMed")
    {
        return 11;
    } else if (muscle == "BFem")
    {
        return 12;
    } else if (muscle == "Semi")
    {
        return 13;
    } else if (muscle == "GlutMax")
    {
        return 14;
    }

}

double MEP_class::getMEP(int gaitCycle, string muscle)
{
    return MEP_class::mep[gaitCycle][getMuscle_Index(muscle)];
}

void getMEP_all(int gaitCycle, double* data)
{
	//memcpy(data, MEP_class::mep[gaitCycle][1], 15);
}

void MEP_class::saveXP()
{
	ostringstream sstream;
	string saveText = "";
		
	cout << "saving XP" << endl;
	for (int i=0; i<4; i++)
	{	
		for (int j=0;j<200; j++)
		{
			sstream << MEP_class::gaussXP[j][i] << ",";
		}
		sstream << endl;
	}
	saveText = sstream.str();
	ofstream saveFile ("save/XP.txt");
	saveFile << saveText; 
	
	saveFile.close();
}

void MEP_class::saveWeightings()
{
	ostringstream sstream;
	string saveText = "";
	
	cout << "saving weigthings" << endl;
	for (int i=0; i<4; i++)
	{	
		for (int j=0;j<15; j++)
		{
			sstream << MEP_class::weightings[j][i] << ",";
		}
		sstream << endl;
	}
	saveText = sstream.str();
	ofstream saveFile ("save/weightings.txt");
	saveFile << saveText; 
	
	saveFile.close();
}

void MEP_class::saveMEP()
{
	ostringstream sstream;
	string saveText = "";
	
	cout << "saving MEP" << endl;
	for (int j=0; j<200; j++)
	{	
		for (int i=0;i<15; i++)
		{
			sstream << MEP_class::mep[j][i] << ",";
		}
		sstream << endl;
	}
	saveText = sstream.str();
	ofstream saveFile ("save/mep.txt");
	saveFile << saveText; 
	
	saveFile.close();
}

void MEP_class::saveData(double data[])
{
	ostringstream sstream;
	string saveText = "";
		
	cout << "saving data" << endl;
	for (int i=0; i<400; i++)
	{	
			sstream << data[i] << ",";
	}

	sstream << endl;
	saveText = sstream.str();
	ofstream saveFile ("save/data_out.txt");
	saveFile << saveText; 
	
	saveFile.close();
}
