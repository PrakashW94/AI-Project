#include "node.h"
#include <math.h>

Node::Node(int id, int type, int inputSize)
{
	nodeId = id;
	nodeType = type;
	bias = 1;
}

int Node::getId()
{
	return nodeId;
}

int Node::getNodeType()
{
	return nodeType;
}

float Node::getBias()
{
	return bias;
}

float Node::getNodeOutput()
{
	return nodeOutput;
}

void Node::setNodeOutput(float value)
{
	if (nodeType == 1)
	{
		nodeOutput = value;
	}
	else
	{
		nodeOutput = 1 / float(1 + exp(-value));
	}
}

float Node::getDelta()
{
	return delta;
}

void Node::setDeltaOutput(float correctValue)
{
	delta = (correctValue - nodeOutput) * (nodeOutput * (1 - nodeOutput));
}

void Node::setDeltaHidden(float weight, float deltaOutput)
{
	delta = weight * deltaOutput * (nodeOutput * (1 - nodeOutput));
}