# =============================================================================
# PINN TRAINING LOOP — FIXED
# =============================================================================
# Copy this into your Colab notebook, replacing the PINN training cell
# (execution_count=35).
#
# Bugs Fixed:
#   - Removed walrus operator (:=) from inside torch.mean()
#   - Split physics loss into clearly named separate variables
#   - Fixed typo: "sensification_loss" → "sensible_heat_loss"
#   - Each loss term is now independently computed, logged, and debuggable
# =============================================================================

# 1. Set hyperparameters and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(piml_model.parameters(), lr=0.01)
epochs = 200

# We need unscaled versions of features inside the loss loop to compute real physics formulas
X_mean = torch.FloatTensor(scaler_X.mean_)
X_std = torch.FloatTensor(scaler_X.scale_)
y_mean = float(scaler_y.mean_[0])
y_std = float(scaler_y.scale_[0])

print("Beginning Physics-Informed training loop...")

for epoch in range(epochs):
    piml_model.train()
    optimizer.zero_grad()

    # Forward pass (scaled outputs)
    predictions_scaled = piml_model(X_train_t)

    # ── DATA LOSS (MSE on scaled values) ──
    loss_data = criterion(predictions_scaled, y_train_t)

    # ── PHYSICS LOSS COMPONENT ──
    # Unscale predictions and inputs to calculate real physical equations
    pred_lst = (predictions_scaled * y_std) + y_mean

    # Extract unscaled features from our training tensor
    # X columns: 0=NDVI, 1=NDWI, 2=Air_Temp, 3=Wind_Speed
    unscaled_X = (X_train_t * X_std) + X_mean
    ndvi_val = unscaled_X[:, 0].unsqueeze(1)
    air_temp_val = unscaled_X[:, 2].unsqueeze(1)
    wind_speed_val = unscaled_X[:, 3].unsqueeze(1)

    # Physics Rule 1: Convective Heat Flux Approximation
    # Surface temperature should strongly scale with Air Temperature under low wind,
    # and deviate predictably as Wind Speed increases.
    surface_air_delta = pred_lst - air_temp_val
    sensible_heat_approx = wind_speed_val * surface_air_delta

    # Physics Rule 2: Evaporative cooling constraint
    # High green cover (NDVI) physically caps how high the surface-to-air delta can go
    # due to transpirational cooling laws.
    physics_violation = torch.relu(surface_air_delta * ndvi_val - 5.0)

    # ── Compute each physics loss term SEPARATELY (Bug Fix) ──
    # Old buggy line was:
    #   loss_physics = torch.mean(torch.abs(sensification_loss := sensible_heat_approx * 0.01)) + torch.mean(physics_violation)
    # Problems: walrus operator inside torch.mean(), typo in variable name,
    #           terms tangled together and impossible to debug individually.

    sensible_heat_loss = torch.mean(torch.abs(sensible_heat_approx * 0.01))
    evaporative_loss = torch.mean(physics_violation)
    loss_physics = sensible_heat_loss + evaporative_loss

    # ── TOTAL LOSS: data + physics ──
    # Adjust lambda (0.1) to balance data accuracy vs physics constraints
    lambda_physics = 0.1
    total_loss = loss_data + (lambda_physics * loss_physics)

    # Backward pass and optimization step
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] -> "
            f"Total: {total_loss.item():.4f} | "
            f"Data MSE: {loss_data.item():.4f} | "
            f"Physics: {loss_physics.item():.4f} "
            f"(Sensible Heat: {sensible_heat_loss.item():.4f}, "
            f"Evaporative: {evaporative_loss.item():.4f})"
        )

print("\n✅ Physics-Informed Neural Network training complete!")
