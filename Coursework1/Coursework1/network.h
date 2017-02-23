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
	vector <vector<float>> accuracyMatrix;

	//constructor
	Network(int inputSize, int hiddenNodesCount);
	//utils
	void outputWeights();
	void forwardPass(vector<float> inputRow);
	void backwardPass(vector<float> inputRow);
	void run(vector<vector<float>> inputData, int passes);
	void runOnce(vector<vector<float>> inputData, int loopCounter, bool createOutput);
	void getOutput(vector<vector<float>> inputData);
	Node getNodeById(int id); //should only be used for read!
	void outputResults();
	void setId(int id);
	void calculateAccuracy();
	void save(string filename);
};