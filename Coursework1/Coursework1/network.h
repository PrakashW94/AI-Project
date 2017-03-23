#pragma once

#include <vector>
#include "node.h"

using namespace std;

class Network 
{
public:
	int networkId;
	int inputNodesCount;
	int hiddenNodesCount;
	unsigned int passes;
	int totalPasses;
	vector<int> kStepPasses;
	vector<float> kFoldsAccuracy;
	float accuracy;
	float testSetAccuracy;
	float rSqrAccuracy;
	float msre;
	float stepParameter = float(0.1);
	vector<Node> nodeList;
	vector<vector<float>> weightsMatrix;
	vector<vector<float>> changeInWeightsMatrix;

	//constructor
	Network(int inputSize, int hiddenNodesCount);
	void outputWeights();
	void kFoldsTraining(vector<vector<vector<float>>> inputDataSet, int networkCount);
	void kFoldsTrainingBD(vector<vector<vector<float>>> inputDataSet, int networkCount);
	void staticTraining(vector<vector<vector<float>>> inputDataSet, int networkCount);
	void staticTrainingBD(vector<vector<vector<float>>> inputDataSet, int networkCount);
	void forwardPass(vector<float> inputRow);
	void backwardPass(vector<float> inputRow, bool momentum = false);
	//void run(vector<vector<float>> inputData, int passes);
	void runOnce(vector<vector<float>> inputData);
	void runBlock(vector<vector<float>> inputData);
	void getOutput(vector<vector<float>> inputData, bool createOutput = false, string fileName = "output2", bool testSet = false);
	Node getNodeById(int id); //should only be used for read!
	void setId(int id);
	void calculateAccuracy(vector<vector<float>> data, bool testSet);
	void calculateRSqrAccuracy(vector<vector<float>> data);
	void calculateMSRE(vector<vector<float>> data);
	void save(string filename);
};