import cv2

# Resim okuma
ref = cv2.imread("/home/ubuntu/Desktop/geyik.png", cv2.IMREAD_COLOR)
tpl = cv2.imread("/home/ubuntu/Desktop/geyikSablon.png", cv2.IMREAD_COLOR)

if ref is None or tpl is None:
    print("Image not found!")
    exit()

# Gri tonlamaya dönüştürme
gref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
gtpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

# SIFT detektör ve özellik çıkarma
sift = cv2.SIFT_create()
keypoints_ref, descriptors_ref = sift.detectAndCompute(gref, None)
keypoints_tpl, descriptors_tpl = sift.detectAndCompute(gtpl, None)

# BFMatcher oluşturma
matcher = cv2.BFMatcher()
matches = matcher.match(descriptors_ref, descriptors_tpl)

# Eşleşmeleri çerçeve içine alma
threshold = 0.2  # Eşik değerini isteğinize göre ayarlayın
good_matches = []
for match in matches:
    if match.distance < threshold * len(descriptors_ref):
        good_matches.append(keypoints_ref[match.queryIdx].pt)

# Çerçeve içine alma
for point in good_matches:
    x, y = point
    rect = (int(x), int(y), tpl.shape[1], tpl.shape[0])
    cv2.rectangle(ref, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

# Çerçeveli resmi gösterme
cv2.imshow("Detected Symbols", ref)
cv2.waitKey(0)

# Kaç tane sembol olduğunu konsola yazdırma
print("Number of Symbols:", len(good_matches))

# Temizleme
cv2.destroyAllWindows()
