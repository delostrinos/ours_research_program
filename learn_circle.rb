#!/usr/bin/env ruby
# frozen_string_literal: true

# Learning the Area of a Circle via LSSR (Least Squares -> Sparsify -> Refine)
# Ruby translation of the MATLAB code from "Learning Mathematics Through the Machine".
#
# The algorithm is NEVER told that area = pi * r^2. It only gets (radius, area)
# pairs -- where each area is computed numerically from an integral -- and it
# discovers that the only meaningful coefficient is the r^2 term, and that it
# equals pi.

require "matrix"

# ----------------------------------------------------------------------------
# Tunable parameters (these mirror numP, numTrials, N in the MATLAB main script)
# ----------------------------------------------------------------------------
NUM_PARAMS  = 11      # degree-10 polynomial => 11 coefficients c0..c10
NUM_TRIALS  = 30      # number of random radii (training samples)
QUAD_POINTS = 100_000 # number of points used to numerically integrate
RADIUS_MIN  = 1.0     # radii sampled uniformly from [RADIUS_MIN, RADIUS_MAX]
RADIUS_MAX  = 5.0

# ----------------------------------------------------------------------------
# Step 1 (INPUT side): the integral that defines the "true" area.
#
#   g(r) = integral from 0 to r of  4 * sqrt(r^2 - x^2) dx
#
# We approximate it with the trapezoidal rule -- exactly like MATLAB's trapz.
# The computer does NOT know this equals pi*r^2; it just sums trapezoids.
# ----------------------------------------------------------------------------
def integrand(r, x)
  4.0 * Math.sqrt([r * r - x * x, 0.0].max) # max(...,0) guards tiny negatives
end

def numeric_area(r, n_points = QUAD_POINTS)
  n = (r * n_points).ceil                # more points for bigger radii
  step = r / (n - 1)                     # spacing h between grid points
  xs = Array.new(n) { |i| i * step }     # grid from 0 to r (like linspace)

  # Trapezoidal rule: 0.5*h*(f0 + 2*f1 + ... + 2*f_{n-2} + f_{n-1})
  total = 0.0
  xs.each_with_index do |x, i|
    weight = (i.zero? || i == n - 1) ? 1.0 : 2.0
    total += weight * integrand(r, x)
  end
  0.5 * step * total
end

# ----------------------------------------------------------------------------
# Helper: solve a least-squares problem  D c ~= b  via the normal equations
#
#   c = (D^T D)^-1 D^T b
#
# This is the textbook formula -- the same one we worked by hand. (See note at
# the bottom about why production code prefers QR/SVD instead.)
# ----------------------------------------------------------------------------
def least_squares(design, observations)
  dt   = design.transpose          # D^T
  dtd  = dt * design               # D^T D   (square matrix)
  dtb  = dt * observations         # D^T b   (column vector)
  dtd.inverse * dtb                # (D^T D)^-1 (D^T b)
end

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

# Generate NUM_TRIALS random radii in [RADIUS_MIN, RADIUS_MAX], then sort
# (sorting is cosmetic, just like the MATLAB version).
radii = Array.new(NUM_TRIALS) { RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * rand }.sort

# Build the observation vector b: the numerically integrated area for each radius.
# (This is MATLAB's `g` -- the INPUT/integral side populating b.)
b = Matrix.column_vector(radii.map { |r| numeric_area(r) })

# Build the Vandermonde matrix D: row i is [1, r, r^2, ..., r^10].
# (This is MATLAB's `R`.)
design = Matrix.build(NUM_TRIALS, NUM_PARAMS) do |i, j|
  radii[i]**j   # j runs 0..10, so columns are r^0 .. r^10
end

# --- L: Least squares -------------------------------------------------------
c_lsq = least_squares(design, b)
puts "Least-squares coefficients (c_lsq), index => value:"
c_lsq.each_with_index do |val, row, _col|
  puts format("  c[%2d] (r^%-2d) = % .8f", row, row, val)
end

# --- S: Sparsify ------------------------------------------------------------
# Threshold = 10% of the largest-magnitude coefficient. Anything smaller is
# treated as numerical noise and forced to zero. (Mirrors MATLAB `sparsify`.)
coeffs = c_lsq.to_a.flatten
max_mag = coeffs.map(&:abs).max
threshold = max_mag / 10.0
survivors = (0...NUM_PARAMS).select { |j| coeffs[j].abs >= threshold }
puts "\nThreshold = #{format('%.6f', threshold)}"
puts "Surviving coefficient indices: #{survivors.inspect}"

# --- R: Refine --------------------------------------------------------------
# Re-solve least squares using ONLY the surviving columns of D. This is the
# honest, simple stand-in for MATLAB's fminunc refinement (see note below).
sub_design = Matrix.build(NUM_TRIALS, survivors.length) do |i, k|
  design[i, survivors[k]]
end
refined = least_squares(sub_design, b)

# Re-embed the refined survivors back into a full-length coefficient vector.
c_star = Array.new(NUM_PARAMS, 0.0)
survivors.each_with_index { |j, k| c_star[j] = refined[k, 0] }

# --- Report -----------------------------------------------------------------
puts "\nFinal sparse coefficients (c_star):"
c_star.each_with_index do |val, j|
  marker = val.abs > 1e-9 ? "  <-- nonzero!" : ""
  puts format("  c[%2d] (r^%-2d) = % .12f%s", j, j, val, marker)
end

learned_pi = c_star[2]
puts "\nLearned coefficient on r^2 : #{format('%.12f', learned_pi)}"
puts "Actual value of pi         : #{format('%.12f', Math::PI)}"
puts "Absolute error             : #{format('%.2e', (learned_pi - Math::PI).abs)}"
