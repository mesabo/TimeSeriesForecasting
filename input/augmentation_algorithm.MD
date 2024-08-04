Function AUGMENT_DATA(cleaned_data):
    Begin
        Initialize augmented_data as [cleaned_data]
        noised_data = add_noise(cleaned_data, noise_std_min=0.1, noise_std_max=1.0)
        scaled_data = scale_data(noised_data, scale_min=0.5, scale_max=2.0)
        permuted_data = permute(scaled_data, spline_order=3, spline_smoothness=20)
        warped_data = warp_data(permuted_data, control_points_min=2, control_points_max=10, warp_factor_min=0.8, warp_factor_max=1.2)
        augmented_data.extend([noised_data, scaled_data, permuted_data, warped_data])
        Return augmented_data
    End
