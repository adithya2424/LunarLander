#include "include/SFML/Graphics.hpp"
#include "include/box2d/box2d.h"
#include <iostream>
#include "include/SFML/Audio.hpp"
#include <cstdlib> 
#include <ctime>


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
            
            std::cout << "Collision detected!" << std::endl;
            collisionDetected = true;
        }
    }
};


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


    void applyProjectedTrajectory(b2World& world, float timeStep, int land_point,int range) {
      b2Vec2 position = body->GetPosition();
    float initialY = position.y;
    float initialX = position.x;

    // std::cout << "Initial position: " << initialX << ", " << initialY << std::endl;

    b2Vec2 finalPosition = position;

    // Simulate the trajectory for a certain time period without modifying the world
    
    for (float t = 0; t <= range; t += timeStep) {
        // Vertical motion equation
        float projectedY = initialY + 20.0f * t - 0.5f * 1.62f * t * t;

        // Horizontal motion equation (adjust the factor as needed)
        float projectedX = initialX + 5.0f * t;

        // Clamp the projectedY value within the bounds of 0 to 600
        projectedY = std::min(std::max(projectedY, 0.0f), 600.0f);

        // Update the final position based on the trajectory
        finalPosition.y = projectedY;
        finalPosition.x = projectedX;
    }

    // Calculate the errors between the current and projected positions
        float errorY = finalPosition.y - position.y;
        float errorX = finalPosition.x - position.x;

        // Apply forces proportional to the errors to move the lander
        float forceMagnitude = 20.0f; // Adjust this value to control the strength of the force
        b2Vec2 force(errorX * forceMagnitude, errorY * forceMagnitude);
        body->ApplyForceToCenter(force, true);

        // Update the physics world for the given time step
        // world.Step(timeStep, 7, 2);
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
    b2Body* groundBody;  // Added member for the ground body in the Box2D world

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

// Rest of your classes remain unchanged
int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "Lunar Lander Simulation");

    sf::SoundBuffer buffer;
    if (!buffer.loadFromFile("/Users/Adithya24/Lunar_Lander_v2/Ali Farka Toure, Ry Cooder - Bonde.wav")) {
        // Handle loading error
    }

    sf::Sound sound;
    sound.setBuffer(buffer);
    bool isPaused = false;
    int land_point = 0;

    b2Vec2 gravity(0.0f, 1.62f); // Moon's gravity (1.62 m/s^2 upwards)
    b2World world(gravity);

    ContactListener contactListener;
    world.SetContactListener(&contactListener);

    Lander lander(world, 50.0f, 50.0f);
    Leg leftLeg(world, 45.0f, 90.0f, {{0.0f, 0.0f}, {10.0f, 0.0f}, {5.0f, 20.0f}, {-5.0f, 20.0f}});
    Leg rightLeg(world, 105.0f, 90.0f, {{0.0f, 0.0f}, {-10.0f, 0.0f}, {-5.0f, 20.0f}, {5.0f, 20.0f}});
    Terrain terrain(world, {{0.0f, 590.0f}, {150.0f, 550.0f}, {220.0f, 440.0f}, {310.0f, 480.0f},
                     {380.0f, 480.0f}, {490.0f, 480.0f}, {700.0f, 490.0f}, {800.0f, 370.0f}});
    Flag leftFlag({{310.0f, 400.0f}, {310.0f, 420.0f}, {330.0f, 410.0f}}, sf::Color::Blue);
    Flag rightFlag({{490.0f, 400.0f}, {490.0f, 420.0f}, {510.0f, 410.0f}}, sf::Color::Blue);
    FlagPole leftFlagPole(310.0f, 480.0f, 310.0f, 400.0f);
    FlagPole rightFlagPole(490.0f, 480.0f, 490.0f, 400.0f);
    
    srand(time(0));
    int range = rand() % 25;
    std::cout << range << std::endl;

    // In your main loop
    bool simulationRunning = true;
    bool successfulLanding = false;
    while (window.isOpen() && simulationRunning) {
    // Check if a collision was detected
    if (contactListener.collisionDetected) {
        std::cout << "Lander crashed!" << std::endl;

        // Add any logic or actions you want to perform when the lander crashes

        // Stop the simulation
        simulationRunning = false;

        // Deactivate the window
        window.setActive(false);

        // Check if the lander is between the flag poles
        b2Vec2 landerPosition = lander.getBody()->GetPosition();
        float landerX = landerPosition.x * 100.0f; // Convert to pixels
        float landerY = landerPosition.y * 100.0f;

        float leftFlagPoleX = leftFlagPole.getX1(); // Replace with the actual method to get X coordinate of the first flag pole
        float rightFlagPoleX = rightFlagPole.getX1(); // Replace with the actual method to get X coordinate of the second flag pole

        if (landerX > leftFlagPoleX && landerX < rightFlagPoleX) {
            std::cout << "Lander successfully landed between the flag poles!" << std::endl;
            // Add any logic or actions you want to perform for a successful landing

            // Display a success message (you can customize this part)
            sf::Font font;
            if (font.loadFromFile("template.ttf")) {
                sf::Text text("Lander Successfully Landed!", font, 40);
                text.setPosition(100, 150);
                text.setFillColor(sf::Color::White);

                while (window.isOpen()) {
                    sf::Event event;
                    while (window.pollEvent(event)) {
                        if (event.type == sf::Event::Closed) {
                            window.close();
                        }
                    }

                    window.clear();
                    window.draw(text);
                    window.display();
                }
            } else {
                std::cerr << "Failed to load font for success message!" << std::endl;
            }
        } else {
            // Display a crashed landing indication (you can customize this part)
            sf::Font font;
            if (font.loadFromFile("template.ttf")) {
                sf::Text text("Lander Crashed!", font, 40);
                text.setPosition(100, 150);
                text.setFillColor(sf::Color::Red);

                while (window.isOpen()) {
                    sf::Event event;
                    while (window.pollEvent(event)) {
                        if (event.type == sf::Event::Closed) {
                            window.close();
                        }
                    }

                    window.clear();
                    window.draw(text);
                    window.display();
                }
            } else {
                std::cerr << "Failed to load font for crashed landing indication!" << std::endl;
            }
        }
    }


        sf::Event event;
        while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    window.close();
                }
                if (event.type == sf::Event::KeyPressed) {
                    if (event.key.code == sf::Keyboard::Space) {
                        sound.play(); // Play the sound when spacebar is pressed
                    }
                    if (event.key.code == sf::Keyboard::LShift) {
                        if (isPaused) {
                            sound.play(); // Resume sound
                            isPaused = false;
                        } else {
                            sound.pause(); // Pause sound
                            isPaused = true;
                        }
                    }
                }
            }
    //land_point = rand(5, 10, 15, 20, 25);
    // Apply the sine wave trajectory to the lander
    
    lander.applyProjectedTrajectory(world, 1.0f / 20000.0f,land_point,range);

    // Get the current lander position
    b2Vec2 landerPosition = lander.getBody()->GetPosition();
    float landerX = landerPosition.x * 100.0f; // Convert to pixels
    float landerY = landerPosition.y * 100.0f;

    // Update the leg positions based on the current lander position
    leftLeg.updatePosition(landerX - 5.0f, landerY + 40.0f);
    rightLeg.updatePosition(landerX + 55.0f, landerY + 40.0f);

    window.clear();

    world.Step(1.0f / 20000.0f, 7, 7);

    lander.draw(window);
    leftLeg.draw(window);
    rightLeg.draw(window);
    terrain.draw(window);
    leftFlag.draw(window);
    rightFlag.draw(window);
    leftFlagPole.draw(window);
    rightFlagPole.draw(window);

    window.display();
}
    return 0;
}
