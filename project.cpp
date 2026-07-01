#include <stdio.h>
#include <math.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "ldmarkmodel.h"

using namespace cv;

// ----------------------------------------------------
// [1] 매크로 상수 및 임계값(Threshold) 정의
// ----------------------------------------------------
#define SMOOTH_ALPHA 0.4f // 스무딩 계수(낮을수록 부드럽지만 반응 느림)
#define PI 3.14159265f

// 표정/자세 판단 기준값
#define THRESH_SMILE_LOW        0.60f  // 미소 비율이 이보다 낮으면 '무표정'
#define THRESH_ASYM_SCORE_HIGH  5.0f   // 비대칭 점수가 이보다 높으면 '입모양 비대칭'
#define THRESH_EYE_CLOSED       0.18f  // EYE (눈 감음 판단)
#define THRESH_EYE_WIDE         0.35f  // EYE (눈 부릅뜸 판단)
#define THRESH_FROWN_SCORE_LOW  0.86f  // 찡그림 점수가 이보다 낮으면 '인상 씀'
#define THRESH_JAW_MOVING       0.45f  // 턱이 이보다 많이 벌어지면 '입 벌림 과다'
#define THRESH_HEAD_ROLL        10.0f  // 고개 기울기가 10도 이상이면 '삐딱함'
#define THRESH_YAW_MIN          0.70f  // Yaw 비율 최소값 (이보다 작으면 한쪽 측면 응시)
#define THRESH_YAW_MAX          1.55f  // Yaw 비율 최대값 (이보다 크면 반대 측면 응시)

#define BUFFER_FRAMES 15

// ----------------------------------------------------
// [2] 데이터 구조체 정의 
// ----------------------------------------------------
typedef struct {
    float eye_avg;       // 눈 뜬 정도
    float jaw_openness;  // 입 벌림 정도 (Jaw Openness)
    float smile_ratio;   // 입꼬리 너비 비율 (Smile Ratio)
    float frown_score;   // 미간 찡그림 점수 (낮을수록 찡그림)
    float roll_angle;    // 고개 좌우 기울기 각도 (Roll)
    float yaw_ratio;     // 고개 회전 비율 (Yaw, 정면=1.0)
    float asym_score;    // 입/턱 비대칭 점수 (높을수록 비대칭)
} FaceFeatures;

typedef struct {
    int yaw_cnt;    // 시선 이탈 카운트
    int roll_cnt;   // 고개 삐딱함 카운트
    int frown_cnt;  // 찡그림 카운트
    int stare_cnt;  // 눈 부릅뜸 카운트
    int blink_cnt;  // 눈 감음/졸음 카운트
    int jaw_cnt;    // 턱 불안정 카운트
    int asym_cnt;   // 비대칭 카운트
    int smile_cnt;  // 무표정 카운트
} WarningCounters;

// ----------------------------------------------------
// [3] 계산 함수
// ----------------------------------------------------

// 두 점 사이의 유클리드 거리 계산
float get_distance(Point2f p1, Point2f p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// 랜드마크 배열에서 (x, y) 좌표 추출
// shape: 랜드마크 데이터 포인터
Point2f get_landmark_point(const Mat* shape, int idx) {
    float x = shape->at<float>(0, idx);
    float y = shape->at<float>(0, idx + 68); // y좌표는 68번 인덱스부터 시작
    return Point2f(x, y);
}

// EYE 눈 깜빡임 계산
// 눈의 가로 길이 대비 세로 길이 비율을 통해 눈이 얼마나 떠져 있는지 계산
float calc_eye(const Mat* shape, int range_start) {
    Point2f p1 = get_landmark_point(shape, range_start + 1); // 윗눈꺼풀 1
    Point2f p5 = get_landmark_point(shape, range_start + 5); // 아랫눈꺼풀 1
    Point2f p2 = get_landmark_point(shape, range_start + 2); // 윗눈꺼풀 2
    Point2f p4 = get_landmark_point(shape, range_start + 4); // 아랫눈꺼풀 2
    Point2f p0 = get_landmark_point(shape, range_start);     // 눈꼬리 (좌)
    Point2f p3 = get_landmark_point(shape, range_start + 3); // 눈꼬리 (우)

    float v1 = get_distance(p1, p5); // 윗눈꺼풀 1과 아랫눈꺼풀 1 사이의 수직 거리 1
    float v2 = get_distance(p2, p4); // 윗눈꺼풀 2과 아랫눈꺼풀 2 사이의수직 거리 2
    float h = get_distance(p0, p3);  // 눈꼬리(좌)와 눈꼬리(우) 사이의 수평 거리

    if (h == 0) return 0.0f; 
    return (v1 + v2) / (2.0f * h);
}

// 두 점의 각도 계산
float calc_angle(Point2f p1, Point2f p2) {
    float dy = p2.y - p1.y;
    float dx = p2.x - p1.x;
    return atan2(dy, dx) * 180.0f / PI;
}

// Yaw (고개 돌림) 비율 계산
// 코(33번)를 중심으로 좌/우 뺨까지의 거리를 비교
// 비율이 1.0에 가까우면 정면, 차이가 크면 측면을 보고 있는 것임
float calc_yaw_ratio(const Mat* shape) {
    Point2f left_cheek = get_landmark_point(shape, 0);  
    Point2f right_cheek = get_landmark_point(shape, 16); 
    Point2f nose_tip = get_landmark_point(shape, 33);    

    float dist_l = fabs(nose_tip.x - left_cheek.x);
    float dist_r = fabs(right_cheek.x - nose_tip.x);
    
    if (dist_r == 0) return 1.0f;
    return dist_l / dist_r; 
}

// Frown (찡그림) 점수 계산
// 미간 너비와 눈썹-눈 사이 거리를 합산. 값이 작아질수록 찡그린 상태
// iod(눈 사이 거리)로 나누어 얼굴 크기에 따른 오차 제거
float calc_frown_score(const Mat* shape, float iod) {
    Point2f brow_inner_l = get_landmark_point(shape, 21); // 왼쪽 눈썹 안쪽
    Point2f brow_inner_r = get_landmark_point(shape, 22); // 오른쪽 눈썹 안쪽
    Point2f eye_inner_l = get_landmark_point(shape, 39);  // 왼쪽 눈 안쪽
    Point2f eye_inner_r = get_landmark_point(shape, 42);  // 오른쪽 눈 안쪽

    float h_squeeze = get_distance(brow_inner_l, brow_inner_r); // 미간 거리
    float v_drop_l = get_distance(brow_inner_l, eye_inner_l);   // 눈썹 높이
    float v_drop_r = get_distance(brow_inner_r, eye_inner_r);

    return (h_squeeze + v_drop_l + v_drop_r) / iod;
}

// Asymmetry (비대칭) 점수 계산
// 입꼬리의 높이 차이, 입의 기울기, 코 중심에서의 거리 차이를 종합 평가
float calc_asym_score(const Mat* shape, float iod, float eye_roll_angle) {
    Point2f mouth_l = get_landmark_point(shape, 48);
    Point2f mouth_r = get_landmark_point(shape, 54);
    Point2f nose = get_landmark_point(shape, 33);
    Point2f eye_l = get_landmark_point(shape, 36);
    Point2f eye_r = get_landmark_point(shape, 45);
    // 1. 각도 차이: 입 기울기가 눈 기울기(얼굴 전체 기울기)와 얼마나 다른지
    float mouth_angle = calc_angle(mouth_l, mouth_r);
    float diff_angle = fabs(eye_roll_angle - mouth_angle);
    // 2. 수직 비대칭: 양쪽 입꼬리 높이 차이
    float dist_v_l = get_distance(eye_l, mouth_l);
    float dist_v_r = get_distance(eye_r, mouth_r);
    float diff_vertical = fabs(dist_v_l - dist_v_r) / iod;
    // 3. 수평 비대칭: 코 중심에서 입꼬리까지 거리 차이
    float dist_h_l = get_distance(nose, mouth_l);
    float dist_h_r = get_distance(nose, mouth_r);
    float diff_horizontal = fabs(dist_h_l - dist_h_r) / iod;

    return diff_angle + (diff_vertical * 50.0f) + (diff_horizontal * 30.0f);
}

// [필터링] 지수 이동 평균
// Jitter 현상을 막기 위해 이전 값과 현재 값을 섞음
void smooth_features(FaceFeatures* current, const FaceFeatures* prev) {
    current->eye_avg = prev->eye_avg * (1.0f - SMOOTH_ALPHA) + current->eye_avg * SMOOTH_ALPHA;
    current->jaw_openness = prev->jaw_openness * (1.0f - SMOOTH_ALPHA) + current->jaw_openness * SMOOTH_ALPHA;
    current->smile_ratio = prev->smile_ratio * (1.0f - SMOOTH_ALPHA) + current->smile_ratio * SMOOTH_ALPHA;
    current->frown_score = prev->frown_score * (1.0f - SMOOTH_ALPHA) + current->frown_score * SMOOTH_ALPHA;
    current->roll_angle = prev->roll_angle * (1.0f - SMOOTH_ALPHA) + current->roll_angle * SMOOTH_ALPHA;
    current->yaw_ratio = prev->yaw_ratio * (1.0f - SMOOTH_ALPHA) + current->yaw_ratio * SMOOTH_ALPHA;
    current->asym_score = prev->asym_score * (1.0f - SMOOTH_ALPHA) + current->asym_score * SMOOTH_ALPHA;
}

// ----------------------------------------------------
// MAIN 함수
// ----------------------------------------------------
int main()
{
    const char* cascade_path = "/opt/homebrew/Cellar/opencv/4.12.0_8/share/opencv4/lbpcascades/lbpcascade_frontalface.xml";
    char model_path[256] = "src/roboman-landmark-model.bin";
    
    CascadeClassifier cascade;
    ldmarkmodel modelt;
    VideoCapture camera(1);
    
    Mat image, gray, current_shape;
    
    FaceFeatures feat;          // 현재 프레임 특징값
    FaceFeatures prev_feat;     // 이전 프레임 특징값 (스무딩용)
    WarningCounters warn_cnt;   // 경고 카운터
    
    int is_reset = 1;       
    int is_head_straight = 0;
    int key = 0;
    int i, idx;             
    int pad = 20;
    
    char message[128];
    char kr_message[128];
    char stats[256];
    Scalar msg_color;

    // 주요 랜드마크 인덱스
    int target_landmarks[] = {
        0, 16, 21, 22, 33, // 얼굴윤곽 시작/끝, 눈썹 안쪽, 코 끝
        36, 37, 38, 39, 40, 41, // 왼쪽 눈 (6개 점)
        42, 43, 44, 45, 46, 47, // 오른쪽 눈 (6개 점)
        48, 54, 51, 57 // 입 (좌우 끝, 위아래 중심)
    };
    int num_targets = sizeof(target_landmarks) / sizeof(target_landmarks[0]);
    
    Point2f eye_l_outer, eye_r_outer;
    float iod; // 동공 간 거리

    memset(&prev_feat, 0, sizeof(FaceFeatures));
    memset(&warn_cnt, 0, sizeof(WarningCounters));

    if (!cascade.load(cascade_path)) {
        if (!cascade.load("lbpcascade_frontalface.xml")) return -1;
    }

    while(!load_ldmarkmodel(model_path, modelt)){
        printf("Model load failed. Input path: ");
        scanf("%s", model_path);
    }

    if(!camera.isOpened()) return 0;

    printf("===== Presentation Coach System (Safe snprintf ver) =====\n");
    printf("Asym Thresh: %.2f | Frown Thresh: %.2f\n", THRESH_ASYM_SCORE_HIGH, THRESH_FROWN_SCORE_LOW);
    printf("Press 'R' to RESET.\n");

    while(1) {
        camera >> image;
        if (image.empty()) break;
        
        // 랜드마크 추적 시도
        modelt.track(image, current_shape);

        // 랜드마크 추적 실패 시 -> OpenCV Cascade로 얼굴 영역 재탐색
        if (current_shape.empty()) {
            cvtColor(image, gray, COLOR_BGR2GRAY);
            std::vector<Rect> faces;
            cascade.detectMultiScale(gray, faces, 1.1, 5, 0, Size(60, 60));
            
            if (!faces.empty()) {
                rectangle(image, faces[0], Scalar(0, 255, 0), 2);
                putText(image, "Detecting...", Point(faces[0].x, faces[0].y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            }
            
            // 얼굴을 놓쳤으므로 카운터 및 상태 리셋
            memset(&warn_cnt, 0, sizeof(WarningCounters));
            is_reset = 1;
        } 
        else {
            // 얼굴 영역 박스 그리기
            float min_x = 10000.0f, max_x = 0.0f, min_y = 10000.0f, max_y = 0.0f;
            int num_pts = current_shape.cols / 2;
            
            for (i = 0; i < num_pts; i++) {
                float x = current_shape.at<float>(0, i);
                float y = current_shape.at<float>(0, i + num_pts);
                if (x < min_x) min_x = x; if (x > max_x) max_x = x;
                if (y < min_y) min_y = y; if (y > max_y) max_y = y;
            }
            rectangle(image, Point((int)min_x - pad, (int)min_y - pad * 2), Point((int)max_x + pad, (int)max_y + pad), Scalar(0, 255, 0), 2);
            
            // IOD 계산
            eye_l_outer = get_landmark_point(&current_shape, 36);
            eye_r_outer = get_landmark_point(&current_shape, 45);
            
            iod = get_distance(eye_l_outer, eye_r_outer);
            if (iod < 1.0f) iod = 1.0f; 

            // 각 부위별 수치화
            feat.eye_avg = (calc_eye(&current_shape, 36) + calc_eye(&current_shape, 42)) / 2.0f;
            feat.smile_ratio = get_distance(get_landmark_point(&current_shape, 48), 
                                            get_landmark_point(&current_shape, 54)) / iod;
            feat.jaw_openness = get_distance(get_landmark_point(&current_shape, 51), 
                                            get_landmark_point(&current_shape, 57)) / iod;
            feat.frown_score = calc_frown_score(&current_shape, iod);
            feat.roll_angle = calc_angle(eye_l_outer, eye_r_outer);
            feat.yaw_ratio = calc_yaw_ratio(&current_shape);
            feat.asym_score = calc_asym_score(&current_shape, iod, feat.roll_angle);

            // 값이 튀는 것을 방지
            if (is_reset) {
                prev_feat = feat;
                is_reset = 0; 
            } else {
                smooth_features(&feat, &prev_feat);
                prev_feat = feat;
            }

            // --------------------------------------------------------
            // 한 번 찡그렸다고 바로 경고하지 않고, BUFFER_FRAMES 동안 지속될 때만 경고
            // --------------------------------------------------------
            
            // 1. 시선 이탈 (Yaw)
            if (feat.yaw_ratio < THRESH_YAW_MIN || feat.yaw_ratio > THRESH_YAW_MAX) warn_cnt.yaw_cnt++;
            else warn_cnt.yaw_cnt = 0;

            // 2. 고개 삐딱함 (Roll)
            if (fabs(feat.roll_angle) > THRESH_HEAD_ROLL) warn_cnt.roll_cnt++;
            else warn_cnt.roll_cnt = 0;

            // 3. 인상 씀 (Frown) - 눈을 뜨고 있는데(졸음 아님) 미간 점수가 낮을 때
            if (feat.eye_avg > THRESH_EYE_CLOSED && feat.frown_score < THRESH_FROWN_SCORE_LOW) warn_cnt.frown_cnt++;
            else warn_cnt.frown_cnt = 0;

            // 4. 눈 부릅뜸 (Stare)
            if (feat.eye_avg > THRESH_EYE_WIDE) warn_cnt.stare_cnt++;
            else warn_cnt.stare_cnt = 0;

            // 5. 졸음/눈감음 (Blink)
            if (feat.eye_avg < THRESH_EYE_CLOSED) warn_cnt.blink_cnt++;
            else warn_cnt.blink_cnt = 0;

            // 6. 턱 움직임 과다 (Jaw)
            if (feat.jaw_openness > THRESH_JAW_MOVING) warn_cnt.jaw_cnt++;
            else warn_cnt.jaw_cnt = 0;

            // 7. 비대칭 (Asym) - 고개를 정면으로 보고 있을 때만 판단
            is_head_straight = (warn_cnt.yaw_cnt == 0);
            if (is_head_straight && feat.asym_score > THRESH_ASYM_SCORE_HIGH) warn_cnt.asym_cnt++;
            else warn_cnt.asym_cnt = 0;

            // 8. 무표정 (Smile Ratio Low)
            if (feat.smile_ratio < THRESH_SMILE_LOW) warn_cnt.smile_cnt++;
            else warn_cnt.smile_cnt = 0;

            // --------------------------------------------------------
            // [결과 시각화 및 메시지 생성]
            // --------------------------------------------------------
            snprintf(message, sizeof(message), "Good Expression :)");
            msg_color = Scalar(0, 255, 0); 

            // 랜드마크 그리기
            for (i = 0; i < num_targets; i++) {
                idx = target_landmarks[i];
                circle(image, get_landmark_point(&current_shape, idx), 3, Scalar(0, 0, 255), -1);
            }

            if (warn_cnt.yaw_cnt > BUFFER_FRAMES) {
                snprintf(message, sizeof(message), "Look at audience");
                msg_color = Scalar(0, 0, 255);
                arrowedLine(image, get_landmark_point(&current_shape, 33), Point(image.cols/2, image.rows/2), msg_color, 2);
            }
            else if (warn_cnt.roll_cnt > BUFFER_FRAMES) {
                snprintf(message, sizeof(message), "Head Straight");
                msg_color = Scalar(0, 0, 255); 
                line(image, eye_l_outer, eye_r_outer, msg_color, 2);
            }
            else if (warn_cnt.frown_cnt > BUFFER_FRAMES) {
                snprintf(message, sizeof(message), "Relax Brows");
                msg_color = Scalar(0, 165, 255); // 주황
                line(image, get_landmark_point(&current_shape, 21), get_landmark_point(&current_shape, 22), msg_color, 2);
            }
            else if (warn_cnt.stare_cnt > BUFFER_FRAMES) {
                snprintf(message, sizeof(message), "Relax Eyes");
                msg_color = Scalar(0, 255, 255); // 노랑
            }
            else if (warn_cnt.blink_cnt > BUFFER_FRAMES) {
                snprintf(message, sizeof(message), "Open Eyes");
                msg_color = Scalar(0, 0, 255);
            }
            else if (warn_cnt.jaw_cnt > BUFFER_FRAMES) {
                snprintf(message, sizeof(message), "Stable Jaw");
                msg_color = Scalar(255, 0, 255); // 보라
            }
            else if (warn_cnt.asym_cnt > BUFFER_FRAMES) {
                snprintf(message, sizeof(message), "Fix Lip Asym");
                msg_color = Scalar(100, 100, 255); // 연한 빨강
                line(image, get_landmark_point(&current_shape, 33), get_landmark_point(&current_shape, 48), msg_color, 1);
                line(image, get_landmark_point(&current_shape, 33), get_landmark_point(&current_shape, 54), msg_color, 1);
                line(image, get_landmark_point(&current_shape, 36), get_landmark_point(&current_shape, 48), msg_color, 1);
                line(image, get_landmark_point(&current_shape, 45), get_landmark_point(&current_shape, 54), msg_color, 1);
            }
            else if (warn_cnt.smile_cnt > BUFFER_FRAMES) {
                snprintf(message, sizeof(message), "Smile :)");
                msg_color = Scalar(200, 200, 200); // 회색
            }

            // 메시지 박스 및 텍스트 출력
            rectangle(image, Point(10, 10), Point(450, 60), Scalar(0, 0, 0), -1); 
            putText(image, message, Point(20, 45), FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2);
        }

        imshow("Presentation Coach", image);

        key = waitKey(5);
        if (key == 27) break; 
        if (key == 'r' || key == 'R') {
            current_shape = Mat(); 
            memset(&warn_cnt, 0, sizeof(WarningCounters)); 
            is_reset = 1;        
            printf("\n\n[SYSTEM] >>> RESET <<<\n");
        }
    }
    printf("\n"); 
    return 0;
}