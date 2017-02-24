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
	int passes;
	float accuracy;
	float stepParameter = float(0.1);
	vector<Node> nodeList;
	vector <vector<float>> weightsMatrix;

	//constructor
	Network(int inputSize, int hiddenNodesCount);
	void outputWeights();
	void selectTraining(int type, vector<vector<float>> inputData, int passes);
	void forwardPass(vector<float> inputRow);
	void backwardPass(vector<float> inputRow);
	//void run(vector<vector<float>> inputData, int passes);
	void runOnce(vector<vector<float>> inputData, int loopCounter, bool createOutput);
	void getOutput(vector<vector<float>> inputData, bool createOutput);
	Node getNodeById(int id); //should only be used for read!
	void outputResults();
	void setId(int id);
	void calculateAccuracy(vector<float> accuracyMatrix);
	void save(string filename);
};