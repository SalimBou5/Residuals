
#ifndef MEP_CLASS
#define MEP_CLASS



class MEP_class
{
	 public:
	//private:
		double regression_coef[15][24];
		double weightings_baseline[4][15];
		
		/* import the data from text files */
		void importRegression_coef();
		void importXP();
		void importBaseline();
		
	//public: 

		double gaussXP[200][4];
		double weightings[4][15];
		double mep[200][15];
		
		MEP_class();
		
		/* calculates the model data for all walking cycle */
		void calcWeightings(double elevation, double speed);
		void calcMEP(double elevation, double speed);
		
        int getMuscle_Index(std::string muscle);
		double getMEP(int gaitCycle, std::string muscle);
		
		/* Gets the pointer to the const for the value (val = 1,2,3),
		 * component (comp = 1,2,3,4), cond ("elevation", "speed") */
		int w_pointer(std::string cond, int comp, int val);
		
		/* Save the output of the model in text files */
		//void saveMEP();
		void saveXP();
		void saveWeightings();
		void saveMEP();
		//void saveWeights();
		void saveData(double data[]);
    
};

#endif
