#include "include/SFML/Graphics.hpp"
#include "include/box2d/box2d.h"
#include <iostream>
#include "include/SFML/Audio.hpp"

class ContactListener : public b2ContactListener {
public:
    void BeginContact(b2Contact* contact) {
        b2Fixture* fixtureA = contact->GetFixtureA();
        b2Fixture* fixtureB = contact->GetFixtureB();

        // Check if either fixture A or B is a terrain fixture
        if (fixtureA->GetUserData().pointer == reinterpret_cast<uintptr_t>("Terrain") ||
            fixtureB->GetUserData().pointer == reinterpret_cast<uintptr_t>("Terrain")) {
            // Handle terrain collision, e.g., play a sound or perform other actions

            // Adjust the lunar lander's properties upon collision
            b2Body* landerBody = nullptr;

            if (fixtureA->GetUserData().pointer == reinterpret_cast<uintptr_t>("Lander")) {
                landerBody = static_cast<b2Body*>(reinterpret_cast<void*>(fixtureA->GetBody()->GetUserData().pointer));
            } else if (fixtureB->GetUserData().pointer == reinterpret_cast<uintptr_t>("Lander")) {
                landerBody = static_cast<b2Body*>(reinterpret_cast<void*>(fixtureB->GetBody()->GetUserData().pointer));
            }

            if (landerBody) {
                // Check if the lander is moving downwards before setting velocity to zero
                b2Vec2 velocity = landerBody->GetLinearVelocity();
                if (velocity.y > 0.0f) {
                    // The lander is moving upwards, no need to set velocity to zero
                    return;
                }

                // Set the linear velocity to zero to make the lander stick to the terrain upon contact
                landerBody->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
            }
        }
    }

    void EndContact(b2Contact* contact) {
        // Handle end contact events if needed
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
        fixtureDef.userData.pointer = reinterpret_cast<uintptr_t>("Lander"); // Updated user data
        body->CreateFixture(&fixtureDef);

        // Store the lunar lander's Box2D body as user data for easy access in collisions
        body->SetUserData(reinterpret_cast<void*>(static_cast<uintptr_t>(reinterpret_cast<uintptr_t>(this))));
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
        body->CreateFixture(&fixtureDef);
    }

    void draw(sf::RenderWindow& window) {
        b2Vec2 position = body->GetPosition();
        shape.setPosition(position.x * 100.0f, position.y * 100.0f); // Convert back to pixels
        window.draw(shape);
    }
};

class Terrain {
private:
    sf::VertexArray shape;
    b2Body* body; // Added member to store Box2D body

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

        // Create Box2D body for terrain
        b2BodyDef bodyDef;
        body = world.CreateBody(&bodyDef);

        // Define the shape of the terrain in Box2D
        b2ChainShape chainShape;
        b2Vec2* vertices = new b2Vec2[points.size()];
        for (size_t i = 0; i < points.size(); ++i) {
            vertices[i].Set(points[i].x / 100.0f, points[i].y / 100.0f); // Convert to meters
        }
        chainShape.CreateChain(vertices, points.size(), b2Vec2(0.0f, 0.0f), b2Vec2(0.0f, 0.0f));
        delete[] vertices;

        // Create fixture for the terrain
        b2FixtureDef fixtureDef;
        fixtureDef.shape = &chainShape;
        fixtureDef.density = 0.0f; // Non-dynamic
        fixtureDef.userData.pointer = reinterpret_cast<uintptr_t>("Terrain"); // Updated user data
        body->CreateFixture(&fixtureDef);
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
};

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "Lunar Lander Simulation");

    sf::SoundBuffer buffer;
    if (!buffer.loadFromFile("/Users/Rohankola/Downloads/CodeProject/Ali Farka Toure, Ry Cooder - Bonde.wav")) {
        // Handle loading error
    }

    sf::Sound sound;
    sound.setBuffer(buffer);
    bool isPaused = false;

    b2Vec2 gravity(0.0f, 1.62f); // Moon's gravity (1.62 m/s^2 upwards)
    b2World world(gravity);

    // Create an instance of the ContactListener
    ContactListener contactListener;

    // Set the contact listener for the Box2D world
    world.SetContactListener(&contactListener);

    Lander lander(world, 50.0f, 50.0f);
    // Store the lunar lander's Box2D body as user data for easy access in collisions
    lander.getBody()->SetUserData(reinterpret_cast<void*>(static_cast<uintptr_t>(reinterpret_cast<uintptr_t>(&lander))));

    Leg leftLeg(world, 45.0f, 90.0f, {{0.0f, 0.0f}, {10.0f, 0.0f}, {5.0f, 20.0f}, {-5.0f, 20.0f}});
    Leg rightLeg(world, 105.0f, 90.0f, {{0.0f, 0.0f}, {-10.0f, 0.0f}, {-5.0f, 20.0f}, {5.0f, 20.0f}});
    Terrain terrain(world, {{0.0f, 590.0f}, {150.0f, 550.0f}, {220.0f, 440.0f}, {310.0f, 480.0f},
                     {380.0f, 480.0f}, {490.0f, 480.0f}, {700.0f, 490.0f}, {800.0f, 370.0f}});
    Flag leftFlag({{310.0f, 400.0f}, {310.0f, 420.0f}, {330.0f, 410.0f}}, sf::Color::Blue);
    Flag rightFlag({{490.0f, 400.0f}, {490.0f, 420.0f}, {510.0f, 410.0f}}, sf::Color::Blue);
    FlagPole leftFlagPole(310.0f, 480.0f, 310.0f, 400.0f);
    FlagPole rightFlagPole(490.0f, 480.0f, 490.0f, 400.0f);

    while (window.isOpen()) {
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

        window.clear();

        world.Step(1.0f / 40000.0f, 7, 2);
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
