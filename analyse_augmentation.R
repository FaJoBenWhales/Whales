data = read.csv("plots/hyperband_2018-02-08-004129_augmentation/table.csv")
LM = lm(final_val_error ~ height_shift_range + width_shift_range + zoom_range + rotation_range, data = data)
# LM = lm(final_val_error ~ height_shift_range + width_shift_range + zoom_range + rotation_range + I(height_shift_range^2) +I( width_shift_range^2) + I(zoom_range^2) + I(rotation_range^2), data = data)
print(summary(LM))