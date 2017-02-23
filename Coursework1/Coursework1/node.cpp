#include "node.h"
#include <math.h>

Node::Node(int id, int type, int inputSize)
{
	nodeId = id;
	nodeType = type;
	bias = 1;
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

void Node::setDeltaOutput(float correctValue)
{
	delta = (correctValue - nodeOutput) * (nodeOutput * (1 - nodeOutput));
}

void Node::setDeltaHidden(float weight, float deltaOutput)
{
	delta = weight * deltaOutput * (nodeOutput * (1 - nodeOutput));
}