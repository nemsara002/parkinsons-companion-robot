import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.putText(
        frame,
        "Parkinson Robot",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

    cv2.rectangle(
    frame,
    (100,100),
    (300,300),
    (255,0,0),
    3
)   
    cv2.imwrite("image.jpg", frame)

    
cap.release()
cv2.destroyAllWindows()