#include "particle.h"

Particle::Particle(cv::Vec3f position)
{
    _position = position;
    _isStopped = false;
}

// Position in 3D space (Slice, X, Y)
cv::Vec3f Particle::position() { return _position; }
// Returns true if particle is stopped
bool Particle::isStopped() { return _isStopped; }
// Sets particle as being stopped
void Particle::stop() { _isStopped = true; }
// Component wise operators
void Particle::operator+=(cv::Vec3f v) { _position += v; }
float Particle::operator()(int index) { return _position(index); }
cv::Vec3f Particle::operator-(Particle p) { return _position - p.position(); }
