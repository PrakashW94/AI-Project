#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include "network.h"

using namespace std;

float randomGen(int n)
{
	int min = int(-20000 / float(n));
	int max = int(20000 / float(n));
	return ((rand() % (max - min + 1) + min) / float(10000));
}

Network::Network(int iNodesCount, int hNodesCount)
{
	networkId = 0;

	inputNodesCount = iNodesCount;
	hiddenNodesCount = hNodesCount;
	accuracy = 1;

	//create input nodes
	//id is 1 to inputNodesCount (inclusive)
	//type is 1
	for (int i = 1; i <= inputNodesCount; i++)
	{
		Node node(i, 1, inputNodesCount);
		nodeList.push_back(node);
	}

	//create hidden nodes
	//id is inputNodesCount + 1 to inputNodesCount + hiddenNodesCount (inclusive)
	//type is 2
	for (int i = inputNodesCount+1; i <= (inputNodesCount + hiddenNodesCount); i++)
	{
		Node node(i, 2, inputNodesCount);
		nodeList.push_back(node);
	}

	//create output node
	//id is inputNodesCount + hiddenNodesCount + 1
	//type is 3
	Node node(inputNodesCount + hiddenNodesCount + 1, 3, 5);
	nodeList.push_back(node);

	//resize outer matrix
	weightsMatrix.resize(inputNodesCount + hiddenNodesCount + 2);

	for (int i = 0; i < inputNodesCount + hiddenNodesCount + 2; i++)
	{
		//resize innter matrix
		//makes access easier as indicies will reflect node connections
		//eg weight in cell [1][6] is weight for input from node 1 to node 6
		weightsMatrix[i].resize(inputNodesCount + hiddenNodesCount + 2);
		//initialise all values to 0
		weightsMatrix[i][0] = 0;
	}

	//populate weights matrix
	//input layer -> hidden layer weights
	for (int i = 1; i <= inputNodesCount; i++)
	{
		for (int j = inputNodesCount + 1; j <= inputNodesCount + hiddenNodesCount; j++)
		{
			weightsMatrix[i][j] = randomGen(inputNodesCount);
		}
	}
	
	//hidden layer -> output layer weights
	for (int i = inputNodesCount + 1; i <= inputNodesCount + hiddenNodesCount; i++)
	{
		weightsMatrix[i][inputNodesCount + hiddenNodesCount + 1] = randomGen(inputNodesCount);
	}

	//bias weights
	for (int i = inputNodesCount + 1; i <= inputNodesCount + hiddenNodesCount + 1; i++)
	{
		weightsMatrix[0][i] = randomGen(inputNodesCount);
	}
}

void Network::kFoldsTraining(vector<vector<vector<float>>> inputDataSet, int desiredPasses, int networkCount, bool boldDriver)
{
	passes = desiredPasses;
	totalPasses = 0;
	float pastAcc;
	int validationSet = 0;
	while (validationSet < 10)
	{
		for (unsigned int i = 1; i <= passes; i++)
		{
			if (i % 100 == 0)
			{//validate
				pastAcc = accuracy;
				getOutput(inputDataSet[validationSet], false);
				if (pastAcc < accuracy)
				{
					passes = i;
				}
			}
			if (i == passes)
			{//last pass, create output files
				kStepPasses.push_back(passes);
				string filename = "kfolds/hn"+ to_string(hiddenNodesCount) + "/n" + to_string(networkCount) + "/vs" + to_string(validationSet);
				getOutput(inputDataSet[validationSet], true, filename);
			}
			else
			{
				for (unsigned int j = 0; j < inputDataSet.size(); j++)
				{
					if (j != validationSet)
					{
						if (boldDriver)
						{//bold driver approach
							bool improved = false;
							int loopBreaker = 0;
							while (!improved && loopBreaker < 5)
							{
								float oldError = accuracy;
								vector<vector<float>> backupWeightsMatrix = weightsMatrix;
								vector<Node> backupNodeList = nodeList;
								runOnce(inputDataSet[j], i, false);
								getOutput(inputDataSet[validationSet]);
								float newError = accuracy;

								if (newError < oldError)
								{
									stepParameter *= 1.1f;
									if (stepParameter > 10) stepParameter = 10;
									improved = true;
									loopBreaker = 0;
								}
								else
								{
									stepParameter *= 0.5;
									if (stepParameter < 0.1) stepParameter = 0.1f;
									weightsMatrix = backupWeightsMatrix;
									nodeList = backupNodeList;
									runOnce(inputDataSet[j], i, false);
									loopBreaker++;
								}
							}
						}
						else
						{//normal approach
							runOnce(inputDataSet[j], i, false);
						}
					}
				}
			}
			totalPasses++;
		}
		validationSet++;
		passes = desiredPasses;
	}
	vector<vector<float>> fullDataSet;
	for (vector<vector<float>> set : inputDataSet)
	{
		for (vector<float> row : set)
		{
			fullDataSet.push_back(row);
		}
	}
	string filename = "kfolds/hn" + to_string(hiddenNodesCount) + "/n" + to_string(networkCount) + "/full";
	getOutput(fullDataSet, true, filename, false);
}

//for each set, loop until you've filled the set
//taking a random index out of the index file in each pass
void Network::staticTraining(vector<vector<vector<float>>> inputDataSet, int desiredPasses, int networkCount, bool boldDriver)
{
	passes = desiredPasses;
	float pastAcc;
	//ofstream accTest;
	//accTest.open("acctest.csv");
	//accTest << "pass,msqer" << endl;
	for (unsigned int i = 1; i <= passes; i++)
	{
		if (i % 100 == 0)
		{//validate
			pastAcc = accuracy;
			getOutput(inputDataSet[1]);
			//accTest << i << ", " << accuracy << endl;
			if (pastAcc < accuracy)
			{
				passes = i;
			}
		}
		if (i == passes)
		{//if last pass, create output files
			string filename = "static/hn" + to_string(hiddenNodesCount) + "/n" +  to_string(networkCount);
			getOutput(inputDataSet[1], true, filename);
			getOutput(inputDataSet[2], true, filename + "ts", true);
		}
		else
		{
			if (boldDriver)
			{//bold driver approach
				bool improved = false;
				int loopBreaker = 0;
				while (!improved && loopBreaker < 5)
				{
					float oldError = accuracy;
					vector<vector<float>> backupWeightsMatrix = weightsMatrix;
					vector<Node> backupNodeList = nodeList;
					runOnce(inputDataSet[0], i, false);
					getOutput(inputDataSet[1]);
					float newError = accuracy;

					if (newError < oldError)
					{
						stepParameter *= 1.1f;
						if (stepParameter > 10) stepParameter = 10;
						improved = true;
						loopBreaker = 0;
					}
					else
					{
						stepParameter *= 0.5;
						if (stepParameter < 0.1) stepParameter = 0.1f;
						weightsMatrix = backupWeightsMatrix;
						nodeList = backupNodeList;
						runOnce(inputDataSet[0], i, false);
						loopBreaker++;
					}
				}
			}//normal approach
			else
			{
				runOnce(inputDataSet[0], i, false);
			}
		}
	}
}

void Network::outputWeights()
{
	cout << "Weights Matrix" << endl;
	for (unsigned int i = 0; i < weightsMatrix.size(); i++)
	{
		cout << "Node " << i << ": ";
		for (unsigned int j = 0; j < weightsMatrix[i].size(); j++)
		{
			cout << weightsMatrix[i][j] << ", ";
		}
		cout << endl;
	}
	cout << endl;
}

Node Network::getNodeById(int id)
{
	return nodeList[id - 1];
}

//for each row in a given dataset, calculate a predicted output and output it
void Network::getOutput(vector<vector<float>> inputData, bool createOutput, string fileName, bool testSet)
{
	vector<float> predictedOutputAccuracy;
	int rowId = 1;

	ofstream outputFile;
	if (createOutput)
	{
		string fullFileName = "output/" + fileName + ".csv";
		outputFile.open(fullFileName);
		outputFile << "Row, Predicted, Correct" << endl;
	}

	for (vector<float> row : inputData)
	{
		forwardPass(row);

		
		int outputNodeId = inputNodesCount + hiddenNodesCount + 1;
		float correctOutput = row.back();
		if (createOutput)
		{//output results
			outputFile << rowId << ", " << getNodeById(outputNodeId).nodeOutput << ", " << correctOutput << endl;
		}
		predictedOutputAccuracy.push_back(correctOutput - getNodeById(outputNodeId).nodeOutput);
		rowId++;
	}
	if (createOutput)
	{
		outputFile.close();
	}
	calculateAccuracy(predictedOutputAccuracy, testSet);
}

//run for a single pass, loop is controlled externally
//allows for different(varying) datasets to be passed in
void Network::runOnce(vector<vector<float>> inputData, int loopCounter, bool createOutput)
{
	int rowId = 1;
	ofstream outputFile;
	if (createOutput)
	{
		outputFile.open("output.csv");
		outputFile << "Loop, Row, Predicted, Correct" << endl;
	}

	for (vector<float> row : inputData)
	{
		forwardPass(row);
		backwardPass(row, true);

		//output results
		if (createOutput)
		{//createOutput is set on the final pass through
			int outputNodeId = inputNodesCount + hiddenNodesCount + 1;
			float correctOutput = row.back();
			outputFile << loopCounter << ", " << rowId << ", " << getNodeById(outputNodeId).nodeOutput << ", " << correctOutput << endl;
		}
		rowId++;
	}
}

//forward pass for single row of data
void Network::forwardPass(vector<float> inputRow)
{
	//set input node values
	for (int i = 1; i <= inputNodesCount; i++)
	{
		nodeList[i - 1].setNodeOutput(inputRow[i - 1]);
	}

	//initialise loop variables
	int inputNodesLower = 1;
	int inputNodesUpper = inputNodesCount;
	int hiddenNodesLower = inputNodesCount + 1;
	int hiddenNodesUpper = inputNodesCount + hiddenNodesCount;
	int outputNodeId = inputNodesCount + hiddenNodesCount + 1;

	//calculate hidden node outputs
	for (int j = hiddenNodesLower; j <= hiddenNodesUpper; j++)
	{
		float hiddenNodeValue = 0;
		for (int i = inputNodesLower; i <= inputNodesUpper; i++)
		{
			hiddenNodeValue += getNodeById(i).nodeOutput * weightsMatrix[i][j];
		}
		hiddenNodeValue += getNodeById(j).bias * weightsMatrix[0][j];
		nodeList[j - 1].setNodeOutput(hiddenNodeValue);
	}

	//calculate output
	float output = 0;
	
	for (int i = hiddenNodesLower; i <= hiddenNodesUpper; i++)
	{
		output += getNodeById(i).nodeOutput * weightsMatrix[i][outputNodeId];
	}
	output += getNodeById(outputNodeId).bias * weightsMatrix[0][outputNodeId];
	nodeList[outputNodeId - 1].setNodeOutput(output);
}
//backward pass for single row of data
void Network::backwardPass(vector<float> inputRow, bool momentum)
{
	//initialise loop variables
	int inputNodesLower = 1;
	int inputNodesUpper = inputNodesCount;
	int hiddenNodesLower = inputNodesCount + 1;
	int hiddenNodesUpper = inputNodesCount + hiddenNodesCount;
	int outputNodeId = inputNodesCount + hiddenNodesCount + 1;
	
	//store correct output
	float correctOutput = inputRow.back();
	
	//backward pass
	//output node
	nodeList[outputNodeId - 1].setDeltaOutput(correctOutput);

	//hidden nodes
	for (int i = hiddenNodesLower; i <= hiddenNodesUpper; i++)
	{
		nodeList[i - 1].setDeltaHidden(weightsMatrix[i][outputNodeId], getNodeById(outputNodeId).delta);
	}

	//update weights
	if (momentum)
	{//with momentum
		float newWeight, deltaWeight;
		float alpha = 0.9f;
		//input layer -> hidden layer
		for (int i = inputNodesLower; i <= inputNodesUpper; i++)
		{
			for (int j = hiddenNodesLower; j <= hiddenNodesUpper; j++)
			{
				deltaWeight = weightsMatrix[i][j];
				newWeight = weightsMatrix[i][j] + (stepParameter * getNodeById(j).delta * getNodeById(i).nodeOutput);
				deltaWeight = newWeight - deltaWeight;
				weightsMatrix[i][j] = newWeight + (alpha * deltaWeight);
			}
		}

		//hidden layer -> output layer
		for (int i = hiddenNodesLower; i <= hiddenNodesUpper; i++)
		{
			deltaWeight = weightsMatrix[i][outputNodeId];
			newWeight = weightsMatrix[i][outputNodeId] + (stepParameter * getNodeById(outputNodeId).delta * getNodeById(i).nodeOutput);
			deltaWeight = newWeight - deltaWeight;
			weightsMatrix[i][outputNodeId] = newWeight + (alpha * deltaWeight);
		}

		//bias weights
		for (int i = hiddenNodesLower; i <= outputNodeId; i++)
		{
			deltaWeight = weightsMatrix[0][i];
			newWeight = weightsMatrix[0][i] + (stepParameter * getNodeById(i).delta * getNodeById(i).bias);
			deltaWeight = newWeight - deltaWeight;
			weightsMatrix[0][i] = newWeight + (alpha * deltaWeight);
		}
	}
	else
	{//without momentum
		//input layer -> hidden layer
		for (int i = inputNodesLower; i <= inputNodesUpper; i++)
		{
			for (int j = hiddenNodesLower; j <= hiddenNodesUpper; j++)
			{
				weightsMatrix[i][j] = weightsMatrix[i][j] + (stepParameter * getNodeById(j).delta * getNodeById(i).nodeOutput);
			}
		}

		//hidden layer -> output layer
		for (int i = hiddenNodesLower; i <= hiddenNodesUpper; i++)
		{
			weightsMatrix[i][outputNodeId] = weightsMatrix[i][outputNodeId] + (stepParameter * getNodeById(outputNodeId).delta * getNodeById(i).nodeOutput);
		}

		//bias weights
		for (int i = hiddenNodesLower; i <= outputNodeId; i++)
		{
			weightsMatrix[0][i] = weightsMatrix[0][i] + (stepParameter * getNodeById(i).delta * getNodeById(i).bias);
		}
	}
}

void Network::calculateAccuracy(vector<float> accuracyMatrix, bool testSet)
{
	float total = 0;
	for (float predictedOutput : accuracyMatrix)
	{
		total += pow(predictedOutput, 2);
	}
	if (testSet)
	{
		testSetAccuracy = total / accuracyMatrix.size();
	}
	else
	{
		accuracy = total / accuracyMatrix.size();
	}
}

void Network::setId(int id)
{
	networkId = id;
}

void Network::save(string filename)
{
	ofstream savefile;
	savefile.open("saved networks/" + filename + ".csv");
	savefile << "iNodes, " << inputNodesCount << endl;
	savefile << "hNodes, " << hiddenNodesCount << endl;
	savefile << "passes, " << passes << endl;
	savefile << "accuracy, " << accuracy << endl;
	for (vector<float> row : weightsMatrix)
	{
		for (float weight : row)
		{
			savefile << weight << ", ";
		}
		savefile << endl;
	}
	savefile.close();
}