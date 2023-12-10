#include "include/SFML/Graphics.hpp"
#include "include/box2d/box2d.h"
#include <iostream>
#include "include/SFML/Audio.hpp"
#include <cstdlib> 
#include <ctime>
#include <thread>
#include <chrono>
#include <atomic>
#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>


float reward = 0.0f; // Initialize the reward variable
bool episode_complete = false;
float crashPenalty = -1.0f; // Penalty for crashing
float landingReward = 1.0f; // Reward for landing between flag poles
float closerToPolesReward = 0.1f; // Penalty for free falling


float BUFFER_SIZE = int(1e5); // replay buffer size
int BATCH_SIZE = 64; // minibatch size
float GAMMA = 0.99; // discount factor
float TAU = 1e-3; // for soft update of target parameters
float LR = 5e-4; // learning rate 
int UPDATE_EVERY = 4; // how often to update the network


// Experience struct for storing experience tuples
struct Experience {
 std::vector<float> state;
 int action;
 float reward;
 std::vector<float> next_state;
 bool done;
};


class ReplayBuffer {
private:
 int action_size;
 int buffer_size;
 int batch_size;
 int seed;

 std::deque<Experience> memory;

public:
 ReplayBuffer(int action_size, int buffer_size, int batch_size, int seed) :
 action_size(action_size),
 buffer_size(buffer_size),
 batch_size(batch_size),
 seed(seed)
 {
 // Initialize other variables and structures here...
 }

 void add(std::vector<float>& state, int action, float reward, std::vector<float>& next_state, bool done) {
 Experience e;
 e.state = state;
 e.action = action;
 e.reward = reward;
 e.next_state = next_state;
 e.done = done;
 memory.push_back(e);
 }

 std::tuple<std::vector<std::vector<float>>, std::vector<int>, std::vector<float>, std::vector<std::vector<float>>, std::vector<bool>> sample() {
 std::vector<std::vector<float>> states;
 std::vector<int> actions;
 std::vector<float> rewards;
 std::vector<std::vector<float>> next_states;
 std::vector<bool> dones;

 std::vector<size_t> indices(memory.size());
 std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., memory.size()-1

 std::random_device rd;
 std::mt19937 gen(rd());
 std::shuffle(indices.begin(), indices.end(), gen); // Shuffle indices


 for (int i = 0; i < batch_size; ++i) {
 int index = indices[i];
 states.push_back(memory[index].state);
 actions.push_back(memory[index].action);
 rewards.push_back(memory[index].reward);
 next_states.push_back(memory[index].next_state);
 dones.push_back(memory[index].done);
 // std::cout << "index, i: " << index << ", " << i << std::endl;
 }

 
 
 return std::make_tuple(states, actions, rewards, next_states, dones);
 }

 int length() {
 return memory.size();
 }
};

// Define the Q-network model
class QNetwork : public torch::nn::Module {
public:

 QNetwork(int inputSize, int outputSize) {
 // Define layers for the neural network
 fc1 = register_module("fc1", torch::nn::Linear(inputSize, 64));
 fc2 = register_module("fc2", torch::nn::Linear(64, 64));
 fc3 = register_module("fc3", torch::nn::Linear(64, outputSize));
 }

 // Define the forward pass through the neural network
 torch::Tensor forward(torch::Tensor x) {
 x = torch::relu(fc1->forward(x));
 x = torch::relu(fc2->forward(x));
 x = fc3->forward(x);
 return x;
 }

 // Define a method to get the number of actions the model outputs
 int getNumActions() const {
 return fc3->weight.size(0); // Accessing the size of weights for output size
 }

private:
 torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
};


class Agent {
private:
 int state_size;
 int action_size;
 int seed;
 ReplayBuffer memory;
 int t_step;

public:
 QNetwork qnetwork_local;
 QNetwork qnetwork_target;
 torch::optim::Adam optimizer;

 Agent(int state_size, int action_size, int seed) :
 state_size(state_size),
 action_size(action_size),
 seed(seed),
 qnetwork_local(state_size, action_size), 
 qnetwork_target(state_size, action_size), 
 memory(action_size, BUFFER_SIZE, BATCH_SIZE, seed),
 optimizer(qnetwork_local.parameters(), torch::optim::AdamOptions(LR))
 {
 t_step = 0;
 }

 void step(std::vector<float>& state, int action, float reward, std::vector<float>& next_state, bool done) {
 // Save experience in replay memory
 memory.add(state, action, reward, next_state, done);
 
 // Learn every UPDATE_EVERY time steps.
 t_step = (t_step + 1) % UPDATE_EVERY;
 if (t_step == 0) {
 // If enough samples are available in memory, get random subset and learn
 if (memory.length() > BATCH_SIZE) {
 // std::cout << "Learning..." << std::endl;
 // std::cout << "memory.length(): " << memory.length() << std::endl;
 auto experiences_tuple = memory.sample();

 // std::cout << "experiences_tuple: " << std::get<0>(experiences_tuple).size() << std::endl;

 // Unpack the tuple into separate vectors
 auto& states = std::get<0>(experiences_tuple);
 auto& actions = std::get<1>(experiences_tuple);
 auto& rewards = std::get<2>(experiences_tuple);
 auto& next_states = std::get<3>(experiences_tuple);
 auto& dones = std::get<4>(experiences_tuple);
 // Convert 'dones' vector<bool> to vector<int>
 std::vector<int> dones_int;
 for (bool done : dones) {
 dones_int.push_back(done ? 1 : 0);
 }

 std::vector<long long> states_size = { static_cast<long long>(states.size()), static_cast<long long>(state_size) };

 
 // std::cout << "states_size: " << states_size[0] << ", " << states_size[1] << std::endl;
 std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> experiences_tensors{
 torch::from_blob(states.data(), { static_cast<long long>(states.size()), static_cast<long long>(state_size) }),
 torch::from_blob(actions.data(), { static_cast<long long>(actions.size()), 1 }),
 torch::from_blob(rewards.data(), { static_cast<long long>(rewards.size()), 1 }),
 torch::from_blob(next_states.data(), { static_cast<long long>(next_states.size()), static_cast<long long>(state_size) }),
 torch::from_blob(dones_int.data(), { static_cast<long long>(dones_int.size()), 1 })
 };

 // std::cout << "experiences_tensors: " << std::get<1>(experiences_tensors).sizes() << std::endl;
 learn(experiences_tensors, GAMMA);
 }
 }
 }

 int act(std::vector<float>& state, float eps = 0.0f) {

 // Convert state to a torch Tensor
 torch::Tensor state_tensor = torch::from_blob(state.data(), {1, state_size}, torch::kFloat32);
 state_tensor.unsqueeze_(0);

 // Set Q-network to evaluation mode and get action values
 qnetwork_local.eval();
 torch::NoGradGuard no_grad;
 torch::Tensor action_values = qnetwork_local.forward(state_tensor);
 qnetwork_local.train();

 // std::cout << "action_values: " << action_values << std::endl;


 // Check if randomly generated number is greater than epsilon
 std::random_device rd;
 std::mt19937 gen(rd());
 std::uniform_real_distribution<float> dis(0.0, 1.0);

 if (dis(gen) > eps) {
 // Convert the tensor to a C++ float array for finding the index of the maximum element
 auto action_values_data = action_values.accessor<float, 3>(); // Assuming shape (1, 1, 5)
 float max_value = action_values_data[0][0][0]; // Initialize with first value
 int max_index = 0;
 
 // Iterate through the tensor to find the maximum value and its index
 for (int i = 0; i < action_size; ++i) {
 if (action_values_data[0][0][i] > max_value) {
 max_value = action_values_data[0][0][i];
 max_index = i;
 }
 }
 // std::cout << "max_index: " << max_index << std::endl;
 return max_index;
 } else {
 // Generate a random index from 0 to (action_size - 1)
 std::uniform_int_distribution<int> int_dis(0, action_size - 1);
 // std::cout << "random_index: " << int_dis(gen) << std::endl;
 return int_dis(gen);
 }
 }

 void learn(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>& experiences, float gamma) {
 // Obtain random minibatch of tuples from D
 auto& states = std::get<0>(experiences);
 auto& actions = std::get<1>(experiences);
 auto& rewards = std::get<2>(experiences);
 auto& next_states = std::get<3>(experiences);
 auto& dones = std::get<4>(experiences);

 // std::cout << "states: " << states << std::endl;


 // std::cout << "states inside learn: " << states << std::endl;

 // Convert actions to a PyTorch tensor
 

 // std::cout << "states: " << states << std::endl;
 // std::cout << "next states: " << next_states << std::endl;

 torch::Tensor qValues = qnetwork_target.forward(next_states);

 // std::cout << "qValues: " << qValues << std::endl;

 // // // // Detach the tensor to prevent gradient tracking if needed
 qValues = qValues.detach();

 // // // // Find the maximum values along dimension 1
 torch::Tensor max_values = std::get<0>(torch::max(qValues, 1));

 // // // // Add a dimension of size 1 at position 1
 torch::Tensor q_targets_next = max_values.unsqueeze(1);

 // std::cout << "max_values_expanded: " << q_targets_next << std::endl;

 // std::cout << "rewards: " << rewards << std::endl;

 // std::cout << "dones: " << dones << std::endl;

 // // // Calculate target value from Bellman equation
 torch::Tensor q_targets = rewards + gamma * q_targets_next * (1 - dones);

 // std::cout << "q_targets: " << q_targets << std::endl;

 // // // Forward pass through the network (replace 'your_network' with your actual network)
 torch::Tensor output = qnetwork_local.forward(states); // Replace 'your_network' with your actual network

 // // // // Perform the gather operation
 torch::Tensor q_expected;

 // // // std::cout << "output: " << output << std::endl;
 // // // std::cout << "actions: " << actions << std::endl;
 actions = actions.to(torch::kInt64);

 // // // // Perform gather along dimension 1 using index 'actions'
 q_expected = output.gather(1, actions);

 // std::cout << "q_expected: " << q_expected << std::endl;

 auto loss = torch::mse_loss(q_expected, q_targets);

 // std::cout << "Loss: " << loss.item<float>() << std::endl;

 optimizer.zero_grad();
 loss.backward();
 optimizer.step();

 // Update target network
 soft_update(qnetwork_local, qnetwork_target, TAU);
 
 }

 void soft_update(QNetwork& local_model, QNetwork& target_model, float tau) {
 // std::cout << "Soft update" << std::endl;
 auto local_params = local_model.parameters();
 auto target_params = target_model.parameters();

 for (size_t i = 0; i < local_params.size(); ++i) {
 target_params[i].data().copy_(tau * local_params[i].data() + (1.0 - tau) * target_params[i].data());
 }

}

};






// Action Space
enum class Action {
 MoveLeft, // Example action: Move left
 MoveRight, // Example action: Move right
 MoveUp, // Example action: Move up
 MoveDown // Example action: Move down
};

// Define the global Box2D world variable
b2World world(b2Vec2(0.0f, 1.62f));

sf::RenderWindow window(sf::VideoMode(800, 600), "Lunar Lander Simulation");


// Define your Agent, Environment, and other necessary classes here
class ContactListener : public b2ContactListener {
public:
 bool collisionDetected = false;
 
 void BeginContact(b2Contact* contact) override {
 // Check if the collision involves the lander legs and the terrain
 b2Fixture* fixtureA = contact->GetFixtureA();
 b2Fixture* fixtureB = contact->GetFixtureB();

 // Check if fixtureA is a lander leg and fixtureB is terrain
 if ((fixtureA->GetFilterData().categoryBits == 0x0004 && fixtureB->GetFilterData().categoryBits == 0x0008) ||
 (fixtureA->GetFilterData().categoryBits == 0x0008 && fixtureB->GetFilterData().categoryBits == 0x0004)) {
 
 // std::cout << "Collision detected!" << std::endl;
 collisionDetected = true;
 }

 }

 // Reset method for ContactListener
 void reset() {
 collisionDetected = false;
 }

};

ContactListener contactListener;


class Lander {
private:
 sf::RectangleShape base;
 b2Body* body;
 

public:
 Lander(b2World& world, float x, float y) {
 base.setSize(sf::Vector2f(50.0f, 50.0f));
 base.setFillColor(sf::Color(128, 0, 128));

 b2BodyDef bodyDef;
 bodyDef.position.Set(x / 100.0f, y / 100.0f); // Convert to meters
 bodyDef.type = b2_dynamicBody;
 body = world.CreateBody(&bodyDef);

 b2PolygonShape shape;
 shape.SetAsBox(25.0f / 100.0f, 25.0f / 100.0f); // Convert to meters

 b2FixtureDef fixtureDef;
 fixtureDef.shape = &shape;
 fixtureDef.density = 1.0f;
 fixtureDef.filter.categoryBits = 0x0002; // Set a unique category bit for the lander
 body->CreateFixture(&fixtureDef);
 }

 void draw(sf::RenderWindow& window) {
 b2Vec2 position = body->GetPosition();
 base.setPosition(position.x * 100.0f, position.y * 100.0f); // Convert back to pixels
 window.draw(base);
 }

 b2Body* getBody() {
 return body;
 }

 sf::RectangleShape& getBase() {
 return base;
 }


 void applyProjectedTrajectory(b2World& world, float timeStep, int range) {
 
 b2Vec2 position = body->GetPosition();
 float initialY = position.y;
 float initialX = position.x;
 b2Vec2 rvelocity = body->GetLinearVelocity();
 float initialVY = rvelocity.y;
 float initialVX = rvelocity.x;

 // std::cout << "Initial position: " << initialX << ", " << initialY << std::endl;

 b2Vec2 finalPosition = position;

 // Simulate the trajectory for a certain time period without modifying the world
 
 for (float t = 0; t <= range; t += timeStep) {
 // Vertical motion equation with initial velocity consideration
 float projectedY = initialY + (initialVY * t) + 20.0f * t - 0.5f * 1.62f * t * t;

 // Horizontal motion equation with initial velocity consideration
 float projectedX = initialX + (initialVX * t) + 5.0f * t;

 // Clamp the projectedY value within the bounds of 0 to 600
 projectedY = std::min(std::max(projectedY, 0.0f), 600.0f);
 projectedX = std::min(std::max(projectedX, 0.0f), 800.0f);

 // Update the final position based on the trajectory
 finalPosition.y = projectedY;
 finalPosition.x = projectedX;
 }

 // Calculate the errors between the current and projected positions
 float errorY = finalPosition.y - position.y;
 float errorX = finalPosition.x - position.x;

 // Apply forces proportional to the errors to move the lander
 float forceMagnitude = 2.0f; // Adjust this value to control the strength of the force
 b2Vec2 force(errorX * forceMagnitude, errorY * forceMagnitude);
 body->ApplyForceToCenter(force, true);

 }


 // Reset method for Lander
 void reset(b2World& world, float x, float y) {
 // Reset the base position and size
 base.setSize(sf::Vector2f(50.0f, 50.0f));
 base.setFillColor(sf::Color(128, 0, 128));
 base.setPosition(x, y);

 // Destroy and re-create the body
 if (body != nullptr) {
 b2World* worldPtr = body->GetWorld();
 worldPtr->DestroyBody(body);
 }

 b2BodyDef bodyDef;
 bodyDef.position.Set(x / 100.0f, y / 100.0f); // Convert to meters
 bodyDef.type = b2_dynamicBody;
 body = world.CreateBody(&bodyDef);

 b2PolygonShape shape;
 shape.SetAsBox(25.0f / 100.0f, 25.0f / 100.0f); // Convert to meters

 b2FixtureDef fixtureDef;
 fixtureDef.shape = &shape;
 fixtureDef.density = 1.0f;
 fixtureDef.filter.categoryBits = 0x0002; // Set a unique category bit for the lander
 body->CreateFixture(&fixtureDef);

 }

void performAction(Action action) {
 // Get the current position of the lander's body
 b2Vec2 currentPosition = body->GetPosition();

 // Define window boundaries
 float windowWidth = 800.0f;
 float windowHeight = 600.0f;

 // Define impulse magnitudes
 float horizontalImpulse = 1.0f; // Adjust as needed
 float verticalImpulse = 1.0f; // Adjust as needed

 // Move the lander based on the action within bounds
 switch (action) {
 case Action::MoveLeft:
 if (currentPosition.x > 0) { // Check if the lander is within the left boundary
 body->ApplyLinearImpulse(b2Vec2(-horizontalImpulse, 0.0f), body->GetWorldCenter(), false);
 }
 else
 {
 body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
 episode_complete = true;
 reward += crashPenalty;
 }
 break;

 case Action::MoveRight:
 // std::cout << "currentPosition.x" << currentPosition.x << std::endl;
 if (currentPosition.x < windowWidth / 100.0f) { // Check if the lander is within the right boundary
 body->ApplyLinearImpulse(b2Vec2(horizontalImpulse, 0.0f), body->GetWorldCenter(), false);
 }
 else
 {
 body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
 episode_complete = true;
 reward += crashPenalty;
 }
 break;

 case Action::MoveUp:
 if (currentPosition.y > 0.1f) { // Check top boundary
 body->ApplyLinearImpulse(b2Vec2(0.0f, -verticalImpulse), body->GetWorldCenter(), true);
 } if (currentPosition.y < 0.1f) {
 // Set velocity to zero when reaching the boundary
 body->SetLinearVelocity(b2Vec2_zero);
 episode_complete = true;
 reward += crashPenalty;
 }
 break;

 case Action::MoveDown:
 if (currentPosition.y < windowHeight / 100.0f) { // Check if the lander is within the bottom boundary
 body->ApplyLinearImpulse(b2Vec2(0.0f, verticalImpulse), body->GetWorldCenter(), false);
 }
 else
 {
 body->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
 episode_complete = true;
 }
 break;

 default:
 std::cout << "Unknown action!\n";
 break;
 }

 }

};

class Leg {
private:
 sf::ConvexShape shape;
 b2Body* body;

public:
 Leg(b2World& world, float x, float y, std::vector<sf::Vector2f> points) {
 shape.setPointCount(points.size());
 for (size_t i = 0; i < points.size(); ++i) {
 shape.setPoint(i, points[i]);
 }
 shape.setPosition(x, y);
 shape.setFillColor(sf::Color(128, 0, 128));

 b2BodyDef bodyDef;
 bodyDef.position.Set(x / 100.0f, y / 100.0f); // Convert to meters
 bodyDef.type = b2_dynamicBody;
 body = world.CreateBody(&bodyDef);

 b2PolygonShape legShape;
 b2Vec2 vertices[4]; // Define the vertices of the leg shape

 // Define the vertices based on the points provided
 for (size_t i = 0; i < points.size(); ++i) {
 vertices[i].Set(points[i].x / 100.0f, points[i].y / 100.0f); // Convert to meters
 }

 legShape.Set(vertices, points.size());

 b2FixtureDef fixtureDef;
 fixtureDef.shape = &legShape;
 fixtureDef.density = 1.0f;
 fixtureDef.filter.categoryBits = 0x0004; // Set a unique category bit for the leg
 body->CreateFixture(&fixtureDef);
 }

 void draw(sf::RenderWindow& window) {
 b2Vec2 position = body->GetPosition();
 shape.setPosition(position.x * 100.0f, position.y * 100.0f); // Convert back to pixels
 window.draw(shape);
 }

 void updatePosition(float x, float y) {
 shape.setPosition(x, y);
 b2Vec2 newPosition = {x / 100.0f, y / 100.0f};
 body->SetTransform(newPosition, body->GetAngle());
 }
};

class Terrain {
private:
 sf::VertexArray shape;
 b2Body* groundBody; // Added member for the ground body in the Box2D world

public:
 Terrain(b2World& world, std::vector<sf::Vector2f> points) {
 shape.setPrimitiveType(sf::LineStrip);
 shape.resize(points.size());
 for (size_t i = 0; i < points.size(); ++i) {
 shape[i].position = points[i];
 }

 for (int i = 0; i < 3; ++i) {
 shape[i].color = sf::Color::Red;
 }

 for (int i = 6; i < 8; ++i) {
 shape[i].color = sf::Color::Red;
 }

 b2BodyDef groundBodyDef;
 groundBodyDef.type = b2_staticBody;
 groundBody = world.CreateBody(&groundBodyDef);

 b2ChainShape groundShape;
 b2Vec2* vertices = new b2Vec2[points.size()];

 for (size_t i = 0; i < points.size(); ++i) {
 vertices[i].Set(points[i].x / 100.0f, points[i].y / 100.0f); // Convert to meters
 }

 groundShape.CreateChain(vertices, points.size(), b2Vec2(0.0f, 0.0f), b2Vec2(0.0f, 0.0f));
 delete[] vertices;

 b2FixtureDef fixtureDef;
 fixtureDef.shape = &groundShape;
 fixtureDef.filter.categoryBits = 0x0008; // Set a unique category bit for the terrain
 groundBody->CreateFixture(&fixtureDef);
 }

 void draw(sf::RenderWindow& window) {
 window.draw(shape);
 }
};

class Flag {
private:
 sf::ConvexShape shape;

public:
 Flag(std::vector<sf::Vector2f> points, sf::Color color) {
 shape.setPointCount(points.size());
 for (size_t i = 0; i < points.size(); ++i) {
 shape.setPoint(i, points[i]);
 }
 shape.setFillColor(color);
 }

 void draw(sf::RenderWindow& window) {
 window.draw(shape);
 }
};

class FlagPole {
private:
 sf::VertexArray shape;

public:
 FlagPole(float x1, float y1, float x2, float y2) {
 shape.setPrimitiveType(sf::Lines);
 shape.resize(2);
 shape[0].position = sf::Vector2f(x1, y1);
 shape[1].position = sf::Vector2f(x2, y2);
 shape[0].color = sf::Color::White;
 shape[1].color = sf::Color::White;
 }

 void draw(sf::RenderWindow& window) {
 window.draw(shape);
 }

 sf::Vector2f getPosition() const {
 return shape[0].position;
 }

 float getX1() const {
 return shape[0].position.x;
 }

 float getY1() const {
 return shape[0].position.y;
 }
};



class LunarLanderEnvironment {
public:
 Lander lander;
 Leg leftLeg;
 Leg rightLeg;
 Terrain terrain;
 Flag leftFlag;
 Flag rightFlag;
 FlagPole leftFlagPole;
 FlagPole rightFlagPole;
 float initialLanderX = 50.0f; // Initial position of the lander on the x-axis
 float initialLanderY = 50.0f; // Initial position of the lander on the y-axis
 float timeStep = 1.0f / 10000.0f; // 60 FPS
 sf::RenderWindow window{sf::VideoMode(800, 600), "Lunar Lander"};
 int state_size = 4;
 int action_size = 4;
 using Reward = float;



public:
 LunarLanderEnvironment() : lander(world, 50.0f, 50.0f),
 leftLeg(world, 45.0f, 90.0f, {{0.0f, 0.0f}, {10.0f, 0.0f}, {5.0f, 20.0f}, {-5.0f, 20.0f}}),
 rightLeg(world, 105.0f, 90.0f, {{0.0f, 0.0f}, {-10.0f, 0.0f}, {-5.0f, 20.0f}, {5.0f, 20.0f}}),
 terrain(world, {{0.0f, 590.0f}, {150.0f, 550.0f}, {220.0f, 440.0f}, {310.0f, 480.0f},
 {380.0f, 480.0f}, {490.0f, 480.0f}, {700.0f, 490.0f}, {800.0f, 370.0f}}),
 leftFlag({{310.0f, 400.0f}, {310.0f, 420.0f}, {330.0f, 410.0f}}, sf::Color::Blue),
 rightFlag({{490.0f, 400.0f}, {490.0f, 420.0f}, {510.0f, 410.0f}}, sf::Color::Blue),
 leftFlagPole(310.0f, 480.0f, 310.0f, 400.0f),
 rightFlagPole(490.0f, 480.0f, 490.0f, 400.0f) {

 

 std::cout << "Lunar Lander Environment created!" << std::endl;
 world.SetContactListener(&contactListener);
 
 

 }

 std::vector<float> reset() {
 window.setActive(true);
 lander.reset(world, 50.0f, 50.0f);
 // Reset the leg positions
 leftLeg.updatePosition(45.0f, 90.0f);
 rightLeg.updatePosition(105.0f, 90.0f);
 episode_complete = false;
 contactListener.reset();
 // Reset the state
 // set the linear velocity to zero
 lander.getBody()->SetLinearVelocity(b2Vec2_zero);
 float initialVY = 0;
 float initialVX = 0;
 std::vector<float> state = {50.0f, 50.0f, initialVX, initialVY};
 reward = 0.0f;
 return state;
 }

 void simulate_freefall(int range) {

 if (contactListener.collisionDetected) {

 // std::cout << "some type of Collision detected!" << std::endl;

 episode_complete = true;

 // Deactivate the window
 window.setActive(false);

 // Check if the lander is between the flag poles
 b2Vec2 landerPosition = lander.getBody()->GetPosition();
 float landerX = landerPosition.x * 100.0f; // Convert to pixels
 float landerY = landerPosition.y * 100.0f;
 float leftFlagPoleX = leftFlagPole.getX1(); // Replace with the actual method to get X coordinate of the first flag pole
 float rightFlagPoleX = rightFlagPole.getX1(); // Replace with the actual method to get X coordinate of the second flag pole
 if (landerX > leftFlagPoleX && landerX < rightFlagPoleX) {

 reward += landingReward;

 // std::cout << "Lander successfully landed between the flag poles!" << std::endl;

 }

 else 
 {
 reward += crashPenalty;

 // std::cout << "Lander crashed!" << std::endl; 
 
 }

 }

 else
 {

 lander.applyProjectedTrajectory(world, timeStep, range);

 // Get the current lander position
 b2Vec2 landerPosition = lander.getBody()->GetPosition();
 float landerX = landerPosition.x * 100.0f; // Convert to pixels
 float landerY = landerPosition.y * 100.0f;

 // Update the leg positions based on the current lander position
 leftLeg.updatePosition(landerX - 5.0f, landerY + 40.0f);
 rightLeg.updatePosition(landerX + 55.0f, landerY + 40.0f);

 // Check if the lander is between the flag poles
 float leftFlagPoleX = leftFlagPole.getX1(); // Replace with the actual method to get X coordinate of the first flag pole
 float rightFlagPoleX = rightFlagPole.getX1(); // Replace with the actual method to get X coordinate of the second flag pole
 if (landerX > leftFlagPoleX && landerX < rightFlagPoleX) {
 // std::cout << "Lander is between the flag poles!" << std::endl;
 reward += closerToPolesReward;
 }

 }

 sf::Event event;
 while (window.pollEvent(event)) {
 if (event.type == sf::Event::Closed) {
 window.close();
 }
 }

 window.clear();
 lander.draw(window);
 leftLeg.draw(window);
 rightLeg.draw(window);
 terrain.draw(window);
 leftFlag.draw(window);
 rightFlag.draw(window);
 leftFlagPole.draw(window);
 rightFlagPole.draw(window);
 
 }

 void render()
 {
 sf::Event event;
 while (window.pollEvent(event)) {
 if (event.type == sf::Event::Closed) {
 window.close();
 }
 }
 window.display();

 }

 Action getActionFromIndex(int index) {
 Action action;
 
 // Map the index to your action space based on the number of actions available
 switch (index) {
 case 0:
 action = Action::MoveLeft;
 break;
 case 1:
 action = Action::MoveRight;
 break;
 case 2:
 action = Action::MoveUp;
 break;
 case 3:
 action = Action::MoveDown;
 break;
 }

 return action;
 }

 std::vector<float> calculateState() {
 b2Vec2 landerPosition = lander.getBody()->GetPosition();
 float landerX = landerPosition.x * 100.0f; // Convert to pixels
 float landerY = landerPosition.y * 100.0f; // Convert to pixels
 b2Vec2 rvelocity = lander.getBody()->GetLinearVelocity();
 float initialVY = rvelocity.y;
 float initialVX = rvelocity.x;

 std::vector<float> current_state = {landerX, landerY, initialVX, initialVY};

 return current_state;
 }

 Reward calculateReward() {
 std::cout << "reward: " << reward << std::endl;
 return reward;
 }

 // RL step function
 std::tuple<std::vector<float>, Reward, bool> step(Action action, int range) {
 simulate_freefall(range); // Call simulateFreefall() function
 lander.performAction(action);
 leftLeg.updatePosition(lander.getBase().getPosition().x - 5.0f, lander.getBase().getPosition().y + 40.0f);
 rightLeg.updatePosition(lander.getBase().getPosition().x + 55.0f, lander.getBase().getPosition().y + 40.0f);
 world.Step(timeStep, 7, 2);
 std::vector<float> new_state = calculateState();
 Reward get_reward = reward;
 bool done = episode_complete; 
 return std::make_tuple(new_state, get_reward, done);
 }

};
// Function to save scores and episode numbers to a CSV file
void saveToCSV(const std::vector<float>& scores, int n_episodes) {
 std::ofstream file("scores_episodes.csv");
 if (file.is_open()) {
 file << "Episode,Score\n";
 for (int i = 0; i < n_episodes; ++i) {
 file << i + 1 << "," << scores[i] << "\n";
 }
 file.close();
 std::cout << "Scores and Episode numbers saved to scores_episodes.csv" << std::endl;
 } else {
 std::cout << "Unable to open file!" << std::endl;
 }
}
// Define your Agent, Environment, and other necessary classes here
std::vector<float> dqn(int n_episodes = 1000, int max_t = 1000, float eps_start = 1.0, float eps_end = 0.01, float eps_decay = 0.995) {
 // Create an instance of the LunarLanderEnvironment
 LunarLanderEnvironment env;
 srand(time(0));
 Agent agent(env.state_size, env.action_size, 0);
 std::vector<float> scores; // List containing scores from each episode
 std::deque<float> scores_window; // Last 100 scores
 float eps = eps_start; // Initialize epsilon
 // env.render(); // Render the environment
 for (int i_episode = 1; i_episode <= n_episodes; ++i_episode) {
 // Assuming 'env' and 'agent' are initialized instances of your Environment and Agent classes
 std::vector<float> state = env.reset();
 // std::cout << "state: \n " << state[0] << " " << state[1] << std::endl;
 // std::cout<< "state full: " << state << std::endl;
 float score = 0;
 int range = rand() % 21;
 for (int t = 0; t < max_t; ++t) {
 // std::cout << "entered here" << std::endl;
 int action = agent.act(state, eps);
 // std::cout << "action: " << action << std::endl;
 Action action_enum = env.getActionFromIndex(action);
 auto [next_state, reward_get, done] = env.step(action_enum, range);
 // std::cout << "reward_get: " << reward_get << std::endl;
 agent.step(state, action, reward_get, next_state, done);
 state = next_state;
 score += reward_get;
 if (done) {
 // std::cout << "breaking out done: " << done << std::endl;
 break;
 }
 env.render();
 }
 scores_window.push_back(score);
 scores.push_back(score);
 eps = std::max(eps_end, eps_decay * eps);

 std::cout << "\rEpisode " << i_episode << "\tAverage Score: " << std::accumulate(scores_window.begin(), scores_window.end(), 0.0f) / scores_window.size() << std::flush;

 if (i_episode % 100 == 0) {
 std::cout << "\rEpisode " << i_episode << "\tAverage Score: " << std::accumulate(scores_window.begin(), scores_window.end(), 0.0f) / scores_window.size() << std::endl;
 }

 }
 std::cout << "Training complete!" << std::endl;
 saveToCSV(scores, n_episodes);

 return scores;
}


int main()
{
 // Start measuring time
 auto start_time = std::chrono::high_resolution_clock::now();
 std::vector<float> scores = dqn(); // Run the DQN training process
 // End measuring time
 auto end_time = std::chrono::high_resolution_clock::now();
 // Calculate the duration (time taken for training)
 auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
 // Print the time taken for training
 std::cout << "Training time: " << duration.count() << " milliseconds" << std::endl;

 return 0;

}
