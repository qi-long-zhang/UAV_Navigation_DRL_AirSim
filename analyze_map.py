import cv2
import numpy as np

def analyze_shapes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    print(f"Image Dimensions: Width={width}, Height={height}")
    
    # Try different thresholds or edges
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500: continue # Ignore noise
        
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            continue
            
        shape_name = "unknown"
        if len(approx) == 3:
            shape_name = "triangle"
        elif len(approx) >= 4:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.7:
                shape_name = "circle"
            else:
                shape_name = f"poly_{len(approx)}"
        
        shapes.append({
            "name": shape_name,
            "center": (cX, cY),
            "area": area
        })
    
    # Filter to get the main shapes (usually they have similar large areas)
    if not shapes:
        print("No shapes detected")
        return
        
    avg_area = sum(s['area'] for s in shapes) / len(shapes)
    main_shapes = [s for s in shapes if s['area'] > avg_area * 0.1]
    
    for s in main_shapes:
        print(f"Shape: {s['name']}, Pos: {s['center']}, Area: {s['area']}")

if __name__ == "__main__":
    analyze_shapes("resources/env_maps/blocks_custom.png")
