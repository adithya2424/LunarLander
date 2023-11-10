#include "include/SFML/Graphics.hpp"
#include "include/box2d/box2d.h"
#include <iostream>
#include "include/SFML/Audio.hpp"


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

public:
    Terrain(std::vector<sf::Vector2f> points) {
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

    b2Vec2 gravity(0.0f, 1.62f); // Moon's gravity (1.62 m/s^2 upwards)
    b2World world(gravity);

    Lander lander(world, 50.0f, 50.0f);
    Leg leftLeg(world, 45.0f, 90.0f, {{0.0f, 0.0f}, {10.0f, 0.0f}, {5.0f, 20.0f}, {-5.0f, 20.0f}});
    Leg rightLeg(world, 105.0f, 90.0f, {{0.0f, 0.0f}, {-10.0f, 0.0f}, {-5.0f, 20.0f}, {5.0f, 20.0f}});
    Terrain terrain({{0.0f, 590.0f}, {150.0f, 550.0f}, {220.0f, 440.0f}, {310.0f, 480.0f},
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

    world.Step(1.0f / 18000.0f, 7, 2);


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
