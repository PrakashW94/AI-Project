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

float calcAnnealedStepParameter(int epochs, int startingEpochs, int hNodes)
{
	float p = 0.01f;
	float q = 0.1f;
	int r = startingEpochs * hNodes;
	float annealedStepParameter = p + (q - p) * (1.0f - (1.0f / (1.0f + exp(10.0f - (20.0f * (float) epochs / (float) r)))));
	return annealedStepParameter;
}

//constructor
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

//kfolds training method
void Network::kFoldsTraining(vector<vector<vector<float>>> inputDataSet, int networkCount)
{
	totalPasses = 0;
	kFoldsAccuracy.resize(10);

	float pastAcc = 1;
	bool trained = false;
	int i = 0;
	while (!trained)
	{
		//train using each fold once
		for (unsigned int validationSet = 0; validationSet < 10; validationSet++)
		{
			for (unsigned int j = 0; j < 10; j++)
			{
				if (j != validationSet)
				{
					runOnce(inputDataSet[j]);
					
				}
			}
			//save accuracy of each validation fold
			getOutput(inputDataSet[validationSet]);
			kFoldsAccuracy[validationSet] = accuracy;
			totalPasses++;
			i++;
		}

		//validate
		if (i % 100 == 0)
		{
			//calculate current accuracy
			accuracy = 0;
			for (float acc : kFoldsAccuracy)
			{
				accuracy += acc;
			}
			accuracy /= 10;
			
			if (pastAcc < accuracy)
			{
				passes = i;
				trained = true;
			}
			else
			{
				pastAcc = accuracy;
			}
		}
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
	getOutput(fullDataSet, true, filename, true);
}

//kfolds training method with bold driver
void Network::kFoldsTrainingBD(vector<vector<vector<float>>> inputDataSet, int networkCount)
{
	ofstream accTest;
	accTest.open("output/exp/acctest" + to_string(networkCount) + ".csv");
	accTest << "epoch, rmse, rmse-dn" << endl;

	ofstream spTest;
	spTest.open("output/exp/sptest" + to_string(networkCount) + ".csv");
	spTest << "epoch, stepParameter" << endl;
	
	totalPasses = 0;
	kFoldsAccuracy.resize(10);
	bool trained = false;
	bool converged = false;
	float pastAcc = 1;
	int epochs = 0;
	float minStepParameter = 0.1f;
	while (!trained)
	{
		if (!converged)
		{
			int convergedCounter = 0;
			//train via bold driver until stepParameter oscilates
			while (!trained && !converged)
			{
				bool improved = false;
				while (!trained && !converged && !improved)
				{
					vector<vector<float>> backupWeightsMatrix = weightsMatrix;
					vector<Node> backupNodeList = nodeList;
					float oldError = accuracy;
					int currentPasses = totalPasses;

					//train for 100 passes
					for (int k = 0; k < 100; k++)
					{
						//train using each fold to validate once
						for (unsigned int validationSet = 0; validationSet < 10; validationSet++)
						{
							for (unsigned int j = 0; j < 10; j++)
							{
								if (j != validationSet)
								{
									runOnce(inputDataSet[j]);
								}
							}
							totalPasses++;
							epochs++;

							//save accuracy of each validation fold on last pass of block
							if (k == 99)
							{
								getOutput(inputDataSet[validationSet]);
								kFoldsAccuracy[validationSet] = accuracy;
							}
						}
						//end training
					}

					//calculate new error
					float newError = 0;
					for (float acc : kFoldsAccuracy)
					{
						newError += acc;
					}
					newError /= 10;

					//check if improvement is made
					if (newError < oldError)
					{
						//if improved
						stepParameter *= 1.1f;
						if (stepParameter > 10) stepParameter = 10;
						accuracy = newError;
						improved = true;

						spTest << epochs << ", " << stepParameter << endl;
					}
					else
					{
						//check whether network has finished training
						if (stepParameter == minStepParameter)
						{
							//no improvement has been made with the minimum stepParameter, network is trained
							trained = true;
						}
						else
						{
							//check whether network has converged and min step rate is too large
							if (stepParameter == minStepParameter*1.1f)
							{
								convergedCounter++;
								if (convergedCounter == 5)
								{
									converged = true;
								}
							}
						}

						//reduce stepParameter to a limit
						stepParameter *= 0.5f;
						if (stepParameter < minStepParameter) stepParameter = minStepParameter;

						//revert network to last step
						weightsMatrix = backupWeightsMatrix;
						nodeList = backupNodeList;
						accuracy = oldError;
						totalPasses = currentPasses;

						spTest << epochs << ", " << stepParameter << endl;
					}
				}

				if (improved)
				{
					if (pastAcc < accuracy)
					{
						passes = epochs;
						trained = true;
					}
					else
					{
						pastAcc = accuracy;
						accTest << epochs << ", " << accuracy << ", " << accuracy*1240.9f << endl;
					}
				}
			}
		}
		else
		{
			//annealing
			int startingEpochs = epochs;
			stepParameter = calcAnnealedStepParameter(epochs, startingEpochs, hiddenNodesCount);
			while (!trained)
			{
				pastAcc = accuracy;

				//train for 100 passes
				for (int k = 0; k < 100; k++)
				{
					//train using each fold to validate once
					for (unsigned int validationSet = 0; validationSet < 10; validationSet++)
					{
						for (unsigned int j = 0; j < 10; j++)
						{
							if (j != validationSet)
							{
								runOnce(inputDataSet[j]);
							}
						}
						totalPasses++;
						epochs++;

						//save accuracy of each validation fold on last pass of block
						if (k == 99)
						{
							getOutput(inputDataSet[validationSet]);
							kFoldsAccuracy[validationSet] = accuracy;
						}
					}
				}
				//end training

				//validate, calculate current accuracy
				accuracy = 0;
				for (float acc : kFoldsAccuracy)
				{
					accuracy += acc;
				}
				accuracy /= 10;

				//check if network has improved	
				if (pastAcc <= accuracy)
				{
					//training is complete
					passes = epochs;
					trained = true;
				}
				else
				{
					//further training required, reduce stepParameter via annealing
					stepParameter = calcAnnealedStepParameter(epochs, startingEpochs, hiddenNodesCount);
					pastAcc = accuracy;

					accTest << epochs << ", " << accuracy << ", " << accuracy*1240.9f << endl;
					spTest << epochs << ", " << stepParameter << endl;
				}
			}
		}
	}

	accTest.close();
	spTest.close();

	vector<vector<float>> fullDataSet;
	for (vector<vector<float>> set : inputDataSet)
	{
		for (vector<float> row : set)
		{
			fullDataSet.push_back(row);
		}
	}
	string filename = "kfolds/hn" + to_string(hiddenNodesCount) + "/n" + to_string(networkCount) + "/full";
	getOutput(fullDataSet, true, filename, true);
}

//for each set, loop until you've filled the set
//taking a random index out of the index file in each pass
void Network::staticTraining(vector<vector<vector<float>>> inputDataSet,  int networkCount)
{
	float pastAcc;
	//ofstream accTest;
	//accTest.open("output/acctest" + to_string(networkCount) + ".csv");
	//accTest << "pass,msqer" << endl;
	int i = 0;
	bool trained = false;
	while(!trained)
	{
		if (i % 100 == 0)
		{//validate
			pastAcc = accuracy;
			getOutput(inputDataSet[1]);
			//accTest << i << ", " << accuracy << endl;
			if (pastAcc < accuracy)
			{
				passes = i;
				trained = true;
			}
		}
		if (trained)
		{//if last pass, create output files
			string filename = "static/hn" + to_string(hiddenNodesCount) + "/n" +  to_string(networkCount);
			getOutput(inputDataSet[1], true, filename);
			getOutput(inputDataSet[2], true, filename + "ts", true);
		}
		else
		{
			runOnce(inputDataSet[0]);	
		}
		i++;
	}
	//accTest.close();
}

void Network::staticTrainingBD(vector<vector<vector<float>>> inputDataSet, int networkCount)
{
	float pastAcc;
	//ofstream accTest;
	//accTest.open("output/acctest" + to_string(networkCount) + ".csv");
	//accTest << "pass,msqer" << endl;
	int i = 0;
	bool trained = false;
	while (!trained)
	{
		pastAcc = accuracy;

		bool improved = false;
		int loopBreaker = 0;
		while (!improved && loopBreaker < 5)
		{
			float oldError = accuracy;
			vector<vector<float>> backupWeightsMatrix = weightsMatrix;
			vector<Node> backupNodeList = nodeList;
			runBlock(inputDataSet[0]);
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
				if (stepParameter == 0.1f) loopBreaker = 5;
				stepParameter *= 0.5;
				if (stepParameter < 0.1) stepParameter = 0.1f;
				weightsMatrix = backupWeightsMatrix;
				nodeList = backupNodeList;
				accuracy = oldError;
				loopBreaker++;
			}
		}
		
		//validate
		
		//accTest << i << ", " << accuracy << endl;
		if (pastAcc <= accuracy)
		{
			passes = i * 100;
			trained = true;
		}
		
		if (trained)
		{//if last pass, create output files
			string filename = "static/hn" + to_string(hiddenNodesCount) + "/n" + to_string(networkCount);
			getOutput(inputDataSet[1], true, filename);
			getOutput(inputDataSet[2], true, filename + "ts", true);
		}
		i++;
	}
	//accTest.close();
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
void Network::runOnce(vector<vector<float>> inputData)
{
	for (vector<float> row : inputData)
	{
		forwardPass(row);
		backwardPass(row, true);
	}
}

//run for 100 passes
void Network::runBlock(vector<vector<float>> inputData)
{
	for (int i = 0; i < 100; i++)
	{
		runOnce(inputData);
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
		testSetAccuracy = sqrt(total / accuracyMatrix.size());
	}
	else
	{
		accuracy = sqrt(total / accuracyMatrix.size());
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
	savefile << "accuracy, " << accuracy << endl << endl;
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