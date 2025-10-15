"""
Test Suite for AdaBoost Implementation

This file contains comprehensive tests for the AdaBoost implementation.
Students can run these tests to verify their implementation is working correctly.

Usage: python test_adaboost.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.tree import DecisionTreeClassifier
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import student's implementation
try:
    from hw3_adaboost_redo import (
        plot_stump_decision, 
        adaboost_round, 
        run_adaboost, 
        plot_ensemble,
        AdaBoostEnsemble
    )
    print("✓ Successfully imported adaboost module")
except ImportError as e:
    print(f"✗ Failed to import adaboost module: {e}")
    sys.exit(1)
except NotImplementedError:
    print("⚠ Module imported but functions not yet implemented")


def generate_test_data():
    """Generate sample datasets for testing."""
    # Simple linearly separable data
    np.random.seed(42)
    X_simple = np.array([
        [1, 2], [2, 3], [3, 1], [2, 1],  # positive class
        [5, 5], [6, 4], [4, 6], [5, 6]   # negative class
    ])
    y_simple = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    # More complex dataset
    X_complex, y_temp = make_classification(
        n_samples=100, 
        n_features=2, 
        n_redundant=0, 
        n_informative=2,
        n_clusters_per_class=1, 
        random_state=42
    )
    y_complex = np.where(y_temp == 0, -1, 1)  # Convert to {-1, 1}

    # Circular dataset (more challenging)
    X_circles, y_temp = make_circles(n_samples=100, noise=0.1, random_state=42)
    y_circles = np.where(y_temp == 0, -1, 1)  # Convert to {-1, 1}

    return (X_simple, y_simple), (X_complex, y_complex), (X_circles, y_circles)


def test_data_format():
    """Test that the data format is correct for AdaBoost."""
    print("\n=== Testing Data Format ===")

    (X_simple, y_simple), (X_complex, y_complex), (X_circles, y_circles) = generate_test_data()

    # Check data shapes
    assert X_simple.shape == (8, 2), f"Expected X_simple shape (8, 2), got {X_simple.shape}"
    assert y_simple.shape == (8,), f"Expected y_simple shape (8,), got {y_simple.shape}"
    assert set(y_simple) == {-1, 1}, f"Expected labels {{-1, 1}}, got {set(y_simple)}"

    print("✓ Data format tests passed")

    
def test_plot_stump_decision(X=None, y=None):
    """Test the plot_stump_decision function."""
    print("\n=== Testing plot_stump_decision ===")

    # Check if exactly one of X or y is None
    if not (X is None and y is None) and (X is None or y is None):
        raise ValueError("X and y must not be None")

    try:
        if X is None and y is None:
            (X_simple, y_simple), _, _ = generate_test_data()
        else:
            X_simple, y_simple = X, y

        # Train a simple stump for testing
        stump = DecisionTreeClassifier(max_depth=1, random_state=42)
        stump.fit(X_simple, y_simple)

        # Test without weights
        fig, ax = plot_stump_decision(X_simple, y_simple, stump)
        assert fig is not None, "Function should return a figure object"
        assert ax is not None, "Function should return an axes object"
        print("✓ plot_stump_decision without weights works")

        # Test with weights
        weights = np.ones(len(y_simple)) / len(y_simple)
        fig, ax = plot_stump_decision(X_simple, y_simple, stump, weights)
        assert fig is not None, "Function should return a figure object with weights"
        print("✓ plot_stump_decision with weights works")

        plt.close('all')  # Clean up

    except NotImplementedError:
        print("⚠ plot_stump_decision not implemented yet")
    except Exception as e:
        print(f"✗ plot_stump_decision test failed: {e}")


def test_adaboost_round():
    """Test the adaboost_round function."""
    print("\n=== Testing adaboost_round ===")

    try:
        (X_simple, y_simple), _, _ = generate_test_data()

        # Initialize uniform weights
        weights = np.ones(len(y_simple)) / len(y_simple)

        # Run one round
        stump, new_weights, model_weight = adaboost_round(X_simple, y_simple, weights)

        # Check outputs
        assert stump is not None, "Should return a trained stump"
        assert isinstance(stump, DecisionTreeClassifier), "Stump should be DecisionTreeClassifier"
        assert new_weights.shape == weights.shape, "New weights should have same shape as input"
        assert abs(new_weights.sum() - 1.0) < 1e-6, "New weights should sum to 1"
        assert model_weight > 0, "Model weight should be positive for a good classifier"

        print("✓ adaboost_round basic functionality works")
        print(f"  - Model weight (alpha): {model_weight:.4f}")
        print(f"  - Weight sum: {new_weights.sum():.6f}")

    except NotImplementedError:
        print("⚠ adaboost_round not implemented yet")
    except Exception as e:
        print(f"✗ adaboost_round test failed: {e}")


def test_run_adaboost():
    """Test the run_adaboost function."""
    print("\n=== Testing run_adaboost ===")

    try:
        (X_simple, y_simple), _, _ = generate_test_data()

        # Run AdaBoost for a few rounds
        r = 3
        stumps, model_weights = run_adaboost(X_simple, y_simple, r)

        # Check outputs
        assert len(stumps) == r, f"Should return {r} stumps, got {len(stumps)}"
        assert len(model_weights) == r, f"Should return {r} model weights, got {len(model_weights)}"
        assert all(isinstance(s, DecisionTreeClassifier) for s in stumps), "All stumps should be DecisionTreeClassifier"
        assert all(w > 0 for w in model_weights), "All model weights should be positive"

        print("✓ run_adaboost basic functionality works")
        print(f"  - Number of stumps: {len(stumps)}")
        print(f"  - Model weights: {[f'{w:.4f}' for w in model_weights]}")

        # Check if figures were saved
        expected_files = [f'figs/round_{i+1}.png' for i in range(r)]
        existing_files = [f for f in expected_files if os.path.exists(f)]
        print(f"  - Saved {len(existing_files)}/{r} figure files")

    except NotImplementedError:
        print("⚠ run_adaboost not implemented yet")
    except Exception as e:
        print(f"✗ run_adaboost test failed: {e}")


# def test_plot_ensemble():
#     """Test the plot_ensemble function (extra credit)."""
#     print("\n=== Testing plot_ensemble (Extra Credit) ===")

#     try:
#         (X_simple, y_simple), _, _ = generate_test_data()

#         # Create a simple ensemble manually for testing
#         stumps = []
#         model_weights = []

#         for i in range(2):
#             stump = DecisionTreeClassifier(max_depth=1, random_state=i)
#             stump.fit(X_simple, y_simple)
#             stumps.append(stump)
#             model_weights.append(0.5)  # Equal weights for simplicity

#         # Test ensemble plotting
#         fig, ax = plot_ensemble(X_simple, y_simple, stumps, model_weights)
#         assert fig is not None, "Function should return a figure object"
#         assert ax is not None, "Function should return an axes object"

#         print("✓ plot_ensemble works (Extra Credit achieved!)")
#         plt.close('all')  # Clean up

#     except NotImplementedError:
#         print("⚠ plot_ensemble not implemented (Extra Credit opportunity)")
#     except Exception as e:
#         print(f"✗ plot_ensemble test failed: {e}")


# def test_ensemble_prediction():
#     """Test ensemble prediction accuracy."""
#     print("\n=== Testing Ensemble Prediction Accuracy ===")

#     try:
#         (X_simple, y_simple), (X_complex, y_complex), _ = generate_test_data()

#         # Test on simple data
#         stumps, model_weights = run_adaboost(X_simple, y_simple, r=5)

#         # Manual ensemble prediction
#         predictions = np.zeros(len(X_simple))
#         for stump, weight in zip(stumps, model_weights):
#             predictions += weight * stump.predict(X_simple)

#         final_predictions = np.sign(predictions)
#         accuracy = np.mean(final_predictions == y_simple)

#         print(f"✓ Simple dataset accuracy: {accuracy:.2%}")

#         # Test on complex data
#         stumps, model_weights = run_adaboost(X_complex, y_complex, r=10)

#         predictions = np.zeros(len(X_complex))
#         for stump, weight in zip(stumps, model_weights):
#             predictions += weight * stump.predict(X_complex)

#         final_predictions = np.sign(predictions)
#         accuracy = np.mean(final_predictions == y_complex)

#         print(f"✓ Complex dataset accuracy: {accuracy:.2%}")

#         if accuracy > 0.7:
#             print("✓ Good ensemble performance achieved!")
#         else:
#             print("⚠ Ensemble performance could be improved")

#     except NotImplementedError:
#         print("⚠ Cannot test ensemble prediction - functions not implemented")
#     except Exception as e:
#         print(f"✗ Ensemble prediction test failed: {e}")


# def test_class_based_approach():
#     """Test the optional class-based implementation."""
#     print("\n=== Testing Class-Based Approach (Optional) ===")

#     try:
#         (X_simple, y_simple), _, _ = generate_test_data()

#         ensemble = AdaBoostEnsemble()
#         # This will only work if students implement the class methods

#         print("⚠ Class-based approach available but not tested (requires student implementation)")

#     except Exception as e:
#         print(f"⚠ Class-based approach not fully implemented: {e}")


def run_visual_demo():
    """Run a visual demonstration of the AdaBoost algorithm."""
    print("\n=== Running Visual Demo ===")

    try:
        # Generate demo data
        (X_simple, y_simple), (X_complex, y_complex), (X_circles, y_circles) = generate_test_data()

        # Demo on different datasets
        datasets = [
            ("Simple Dataset", X_simple, y_simple, 3),
            ("Complex Dataset", X_complex, y_complex, 5),
            ("Circles Dataset", X_circles, y_circles, 10)
        ]

        for name, X, y, rounds in datasets:
            print(f"\n--- {name} ---")
            try:
                stumps, weights = run_adaboost(X, y, rounds)
                print(f"✓ Completed {rounds} rounds of AdaBoost")

                # Try extra credit ensemble plot
                try:
                    fig, ax = plot_ensemble(X, y, stumps, weights)
                    plt.title(f"AdaBoost Ensemble - {name}")
                    plt.savefig(f'figs/ensemble_{name.lower().replace(" ", "_")}.png')
                    plt.close()
                    print("✓ Saved ensemble plot")
                except NotImplementedError:
                    print("⚠ Ensemble plotting not implemented")

            except NotImplementedError:
                print(f"⚠ Cannot demo {name} - core functions not implemented")
            except Exception as e:
                print(f"✗ Demo failed for {name}: {e}")

    except Exception as e:
        print(f"✗ Visual demo failed: {e}")


def main():
    """Run all tests."""
    print("AdaBoost Implementation Test Suite")
    print("=" * 50)

    # Create output directory
    os.makedirs('figs', exist_ok=True)

    # Run all tests
    test_data_format()
    test_plot_stump_decision()
    test_adaboost_round()
    test_run_adaboost()
    # test_plot_ensemble()
    # test_ensemble_prediction()
    # test_class_based_approach()

    # Run visual demo
    run_visual_demo()

    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nInstructions for students:")
    print("1. Implement each function in adaboost.py")
    print("2. Run this test file to verify your implementation")
    print("3. Check the 'figs' directory for generated plots")
    print("4. Implement plot_ensemble() for extra credit")
    print("\nGood luck with your implementation! 🚀")


if __name__ == "__main__":
    main()