#include <GL/glew.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <eigen3/Eigen/Dense>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <stdarg.h>
#include <time.h>

using namespace Eigen;

/* Options */

#ifdef HIGH_QUALITY
enum { WINDOW_WIDTH  = 1280, WINDOW_HEIGHT = 720 };
#else
enum { WINDOW_WIDTH  = 720, WINDOW_HEIGHT = 480 };
#endif

enum {
  GAMEPAD_BACK = 5,
  GAMEPAD_START = 4,
  GAMEPAD_A = 10,
  GAMEPAD_B = 11,
  GAMEPAD_X = 12,
  GAMEPAD_Y = 13,
  GAMEPAD_TRIGGER_L  = 4,
  GAMEPAD_TRIGGER_R  = 5,
  GAMEPAD_SHOULDER_L = 8,
  GAMEPAD_SHOULDER_R = 9,
  GAMEPAD_STICK_L_HORIZONTAL = 0,
  GAMEPAD_STICK_L_VERTICAL   = 1,
  GAMEPAD_STICK_R_HORIZONTAL = 2,
  GAMEPAD_STICK_R_VERTICAL   = 3
};


/* Phase-Functioned Neural Network */

struct PFNN {
  // TODO 3: 修改输入维度
  // enum { XDIM = 342, YDIM = 311, HDIM = 512 };
  enum { XDIM = 276, YDIM = 212, HDIM = 512 };

  enum { MODE_CONSTANT, MODE_LINEAR, MODE_CUBIC };

  int mode;
  
  ArrayXf Xmean, Xstd;
  ArrayXf Ymean, Ystd;
  
  std::vector<ArrayXXf> W0, W1, W2;
  std::vector<ArrayXf>  b0, b1, b2;
  
  ArrayXf  Xp, Yp;
  ArrayXf  H0,  H1;
  ArrayXXf W0p, W1p, W2p;
  ArrayXf  b0p, b1p, b2p;
   
  PFNN(int pfnnmode)
    : mode(pfnnmode) { 
    
    Xp = ArrayXf((int)XDIM);
    Yp = ArrayXf((int)YDIM);
    
    H0 = ArrayXf((int)HDIM);
    H1 = ArrayXf((int)HDIM);
    
    W0p = ArrayXXf((int)HDIM, (int)XDIM);
    W1p = ArrayXXf((int)HDIM, (int)HDIM);
    W2p = ArrayXXf((int)YDIM, (int)HDIM);
    
    b0p = ArrayXf((int)HDIM);
    b1p = ArrayXf((int)HDIM);
    b2p = ArrayXf((int)YDIM);
  }
  
  static void load_weights(ArrayXXf &A, int rows, int cols, const char* fmt, ...) {
    va_list valist;
    va_start(valist, fmt);
    char filename[512];
    vsprintf(filename, fmt, valist);
    va_end(valist);

    FILE *f = fopen(filename, "rb");
    if (f == NULL) { fprintf(stderr, "Couldn't load file %s\n", filename); exit(1); }

    A = ArrayXXf(rows, cols);
    for (int x = 0; x < rows; x++)
    for (int y = 0; y < cols; y++) {
      float item = 0.0;
      fread(&item, sizeof(float), 1, f);
      A(x, y) = item;
    }
    fclose(f); 
  }

  static void load_weights(ArrayXf &V, int items, const char* fmt, ...) {
    va_list valist;
    va_start(valist, fmt);
    char filename[512];
    vsprintf(filename, fmt, valist);
    va_end(valist);
    
    FILE *f = fopen(filename, "rb"); 
    if (f == NULL) { fprintf(stderr, "Couldn't load file %s\n", filename); exit(1); }
    
    V = ArrayXf(items);
    for (int i = 0; i < items; i++) {
      float item = 0.0;
      fread(&item, sizeof(float), 1, f);
      V(i) = item;
    }
    fclose(f); 
  }  
  
  void load() {
    
    load_weights(Xmean, XDIM, "./network/pfnn/Xmean.bin");
    load_weights(Xstd,  XDIM, "./network/pfnn/Xstd.bin");
    load_weights(Ymean, YDIM, "./network/pfnn/Ymean.bin");
    load_weights(Ystd,  YDIM, "./network/pfnn/Ystd.bin");
    
    switch (mode) {
      
      case MODE_CONSTANT:
        
        W0.resize(50); W1.resize(50); W2.resize(50);
        b0.resize(50); b1.resize(50); b2.resize(50);
      
        for (int i = 0; i < 50; i++) {            
          load_weights(W0[i], HDIM, XDIM, "./network/pfnn/W0_%03i.bin", i);
          load_weights(W1[i], HDIM, HDIM, "./network/pfnn/W1_%03i.bin", i);
          load_weights(W2[i], YDIM, HDIM, "./network/pfnn/W2_%03i.bin", i);
          load_weights(b0[i], HDIM, "./network/pfnn/b0_%03i.bin", i);
          load_weights(b1[i], HDIM, "./network/pfnn/b1_%03i.bin", i);
          load_weights(b2[i], YDIM, "./network/pfnn/b2_%03i.bin", i);            
        }
        
      break;
      
      {

      // case MODE_LINEAR:
      
      //   W0.resize(10); W1.resize(10); W2.resize(10);
      //   b0.resize(10); b1.resize(10); b2.resize(10);
      
      //   for (int i = 0; i < 10; i++) {
      //     load_weights(W0[i], HDIM, XDIM, "./network/pfnn/W0_%03i.bin", i * 5);
      //     load_weights(W1[i], HDIM, HDIM, "./network/pfnn/W1_%03i.bin", i * 5);
      //     load_weights(W2[i], YDIM, HDIM, "./network/pfnn/W2_%03i.bin", i * 5);
      //     load_weights(b0[i], HDIM, "./network/pfnn/b0_%03i.bin", i * 5);
      //     load_weights(b1[i], HDIM, "./network/pfnn/b1_%03i.bin", i * 5);
      //     load_weights(b2[i], YDIM, "./network/pfnn/b2_%03i.bin", i * 5);  
      //   }
      
      // break;
      
      // case MODE_CUBIC:
      
      //   W0.resize(4); W1.resize(4); W2.resize(4);
      //   b0.resize(4); b1.resize(4); b2.resize(4);
      
      //   for (int i = 0; i < 4; i++) {
      //     load_weights(W0[i], HDIM, XDIM, "./network/pfnn/W0_%03i.bin", (int)(i * 12.5));
      //     load_weights(W1[i], HDIM, HDIM, "./network/pfnn/W1_%03i.bin", (int)(i * 12.5));
      //     load_weights(W2[i], YDIM, HDIM, "./network/pfnn/W2_%03i.bin", (int)(i * 12.5));
      //     load_weights(b0[i], HDIM, "./network/pfnn/b0_%03i.bin", (int)(i * 12.5));
      //     load_weights(b1[i], HDIM, "./network/pfnn/b1_%03i.bin", (int)(i * 12.5));
      //     load_weights(b2[i], YDIM, "./network/pfnn/b2_%03i.bin", (int)(i * 12.5));  
      //   }
        
      // break;

      }

    }
    
  }
  
  static void ELU(ArrayXf &x) { x = x.max(0) + x.min(0).exp() - 1; }

  static void linear(ArrayXf  &o, const ArrayXf  &y0, const ArrayXf  &y1, float mu) { o = (1.0f-mu) * y0 + (mu) * y1; }
  static void linear(ArrayXXf &o, const ArrayXXf &y0, const ArrayXXf &y1, float mu) { o = (1.0f-mu) * y0 + (mu) * y1; }
  
  static void cubic(ArrayXf  &o, const ArrayXf &y0, const ArrayXf &y1, const ArrayXf &y2, const ArrayXf &y3, float mu) {
    o = (
      (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
      (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
      (-0.5*y0+0.5*y2)*mu + 
      (y1));
  }
  
  static void cubic(ArrayXXf &o, const ArrayXXf &y0, const ArrayXXf &y1, const ArrayXXf &y2, const ArrayXXf &y3, float mu) {
    o = (
      (-0.5*y0+1.5*y1-1.5*y2+0.5*y3)*mu*mu*mu + 
      (y0-2.5*y1+2.0*y2-0.5*y3)*mu*mu + 
      (-0.5*y0+0.5*y2)*mu + 
      (y1));
  }

  void predict(float P) {
    
    float pamount;
    int pindex_0, pindex_1, pindex_2, pindex_3;
    
    Xp = (Xp - Xmean) / Xstd;
    
    switch (mode) {
      
      case MODE_CONSTANT:
        pindex_1 = (int)((P / (2*M_PI)) * 50);
        H0 = (W0[pindex_1].matrix() * Xp.matrix()).array() + b0[pindex_1]; ELU(H0);
        H1 = (W1[pindex_1].matrix() * H0.matrix()).array() + b1[pindex_1]; ELU(H1);
        Yp = (W2[pindex_1].matrix() * H1.matrix()).array() + b2[pindex_1];
      break;
      
      {

      // case MODE_LINEAR:
      //   pamount = fmod((P / (2*M_PI)) * 10, 1.0);
      //   pindex_1 = (int)((P / (2*M_PI)) * 10);
      //   pindex_2 = ((pindex_1+1) % 10);
      //   linear(W0p, W0[pindex_1], W0[pindex_2], pamount);
      //   linear(W1p, W1[pindex_1], W1[pindex_2], pamount);
      //   linear(W2p, W2[pindex_1], W2[pindex_2], pamount);
      //   linear(b0p, b0[pindex_1], b0[pindex_2], pamount);
      //   linear(b1p, b1[pindex_1], b1[pindex_2], pamount);
      //   linear(b2p, b2[pindex_1], b2[pindex_2], pamount);
      //   H0 = (W0p.matrix() * Xp.matrix()).array() + b0p; ELU(H0);
      //   H1 = (W1p.matrix() * H0.matrix()).array() + b1p; ELU(H1);
      //   Yp = (W2p.matrix() * H1.matrix()).array() + b2p;
      // break;
      
      // case MODE_CUBIC:
      //   pamount = fmod((P / (2*M_PI)) * 4, 1.0);
      //   pindex_1 = (int)((P / (2*M_PI)) * 4);
      //   pindex_0 = ((pindex_1+3) % 4);
      //   pindex_2 = ((pindex_1+1) % 4);
      //   pindex_3 = ((pindex_1+2) % 4);
      //   cubic(W0p, W0[pindex_0], W0[pindex_1], W0[pindex_2], W0[pindex_3], pamount);
      //   cubic(W1p, W1[pindex_0], W1[pindex_1], W1[pindex_2], W1[pindex_3], pamount);
      //   cubic(W2p, W2[pindex_0], W2[pindex_1], W2[pindex_2], W2[pindex_3], pamount);
      //   cubic(b0p, b0[pindex_0], b0[pindex_1], b0[pindex_2], b0[pindex_3], pamount);
      //   cubic(b1p, b1[pindex_0], b1[pindex_1], b1[pindex_2], b1[pindex_3], pamount);
      //   cubic(b2p, b2[pindex_0], b2[pindex_1], b2[pindex_2], b2[pindex_3], pamount);
      //   H0 = (W0p.matrix() * Xp.matrix()).array() + b0p; ELU(H0);
      //   H1 = (W1p.matrix() * H0.matrix()).array() + b1p; ELU(H1);
      //   Yp = (W2p.matrix() * H1.matrix()).array() + b2p;
      // break;
      
      }
      
      default:
      break;
    }
    
    Yp = (Yp * Ystd) + Ymean;

  }
  
  
  
};

static PFNN* pfnn = NULL;

/* Joystick */

static SDL_Joystick* stick = NULL;

/* Heightmap */

struct Heightmap {
  
  float hscale;
  float vscale;
  float offset;
  std::vector<std::vector<float>> data;
  GLuint vbo;
  GLuint tbo;
  
  Heightmap()
    : hscale(3.937007874)
    //, vscale(3.937007874)
    , vscale(3.0)
    , offset(0.0)
    , vbo(0)
    , tbo(0) {}
  
  ~Heightmap() {
    if (vbo != 0) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (tbo != 0) { glDeleteBuffers(1, &tbo); tbo = 0; } 
  }
  
  void load(const char* filename, float multiplier) {
    
    vscale = multiplier * vscale;
    
    if (vbo != 0) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (tbo != 0) { glDeleteBuffers(1, &tbo); tbo = 0; }
    
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &tbo);
    
    data.clear();
    
    std::ifstream file(filename);
    
    std::string line;
    while (std::getline(file, line)) {
      std::vector<float> row;
      std::istringstream iss(line);
      while (iss) {
        float f;
        iss >> f;
        row.push_back(f);
      }
      data.push_back(row);
    }
    
    int w = data.size();
    int h = data[0].size();
    
    offset = 0.0;
    for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) {
      offset += data[x][y];
    }
    offset /= w * h;
    
    printf("Loaded Heightmap '%s' (%i %i)\n", filename, (int)w, (int)h);
    
    glm::vec3* posns = (glm::vec3*)malloc(sizeof(glm::vec3) * w * h);
    glm::vec3* norms = (glm::vec3*)malloc(sizeof(glm::vec3) * w * h);
    float* aos   = (float*)malloc(sizeof(float) * w * h);
    
    for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) {
      float cx = hscale * x, cy = hscale * y, cw = hscale * w, ch = hscale * h;
      posns[x+y*w] = glm::vec3(cx - cw/2, sample(glm::vec2(cx-cw/2, cy-ch/2)), cy - ch/2);
    }
    
    for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) {
      norms[x+y*w] = (x > 0 && x < w-1 && y > 0 && y < h-1) ?
        glm::normalize(glm::mix(
          glm::cross(
            posns[(x+0)+(y+1)*w] - posns[x+y*w],
            posns[(x+1)+(y+0)*w] - posns[x+y*w]),
          glm::cross(
            posns[(x+0)+(y-1)*w] - posns[x+y*w],
            posns[(x-1)+(y+0)*w] - posns[x+y*w]), 0.5)) : glm::vec3(0,1,0);
    }


    char ao_filename[512];
    memcpy(ao_filename, filename, strlen(filename)-4);
    ao_filename[strlen(filename)-4] = '\0';
    strcat(ao_filename, "_ao.txt");
    
    srand(0);

    FILE* ao_file = fopen(ao_filename, "r");
    bool ao_generate = false;
    if (ao_file == NULL || ao_generate) {
      ao_file = fopen(ao_filename, "w");
      //ao_generate = true;
    }
   
    for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) {
      
      if (ao_generate) {
      
        float ao_amount = 0.0;
        float ao_radius = 50.0;
        int ao_samples = 1024;
        int ao_steps = 5;
        for (int i = 0; i < ao_samples; i++) {
          glm::vec3 off = glm::normalize(glm::vec3(rand() % 10000 - 5000, rand() % 10000 - 5000, rand() % 10000 - 5000));
          if (glm::dot(off, norms[x+y*w]) < 0.0f) { off = -off; }
          for (int j = 1; j <= ao_steps; j++) {
            glm::vec3 next = posns[x+y*w] + (((float)j) / ao_steps) * ao_radius * off;
            if (sample(glm::vec2(next.x, next.z)) > next.y) { ao_amount += 1.0; break; }
          }
        }
        
        aos[x+y*w] = 1.0 - (ao_amount / ao_samples);
        fprintf(ao_file, y == h-1 ? "%f\n" : "%f ", aos[x+y*w]);
      } else {
        fscanf(ao_file, y == h-1 ? "%f\n" : "%f ", &aos[x+y*w]);
      }
      
    }
    
    fclose(ao_file);

    float *vbo_data = (float*)malloc(sizeof(float) * 7 * w * h);
    uint32_t *tbo_data = (uint32_t*)malloc(sizeof(uint32_t) * 3 * 2 * (w-1) * (h-1));
    
    for (int x = 0; x < w; x++)
    for (int y = 0; y < h; y++) {
      vbo_data[x*7+y*7*w+0] = posns[x+y*w].x; 
      vbo_data[x*7+y*7*w+1] = posns[x+y*w].y;
      vbo_data[x*7+y*7*w+2] = posns[x+y*w].z;
      vbo_data[x*7+y*7*w+3] = norms[x+y*w].x;
      vbo_data[x*7+y*7*w+4] = norms[x+y*w].y;
      vbo_data[x*7+y*7*w+5] = norms[x+y*w].z; 
      vbo_data[x*7+y*7*w+6] = aos[x+y*w]; 
    }
    
    free(posns);
    free(norms);
    free(aos);
    
    for (int x = 0; x < (w-1); x++)
    for (int y = 0; y < (h-1); y++) {
      tbo_data[x*3*2+y*3*2*(w-1)+0] = (x+0)+(y+0)*w;
      tbo_data[x*3*2+y*3*2*(w-1)+1] = (x+0)+(y+1)*w;
      tbo_data[x*3*2+y*3*2*(w-1)+2] = (x+1)+(y+0)*w;
      tbo_data[x*3*2+y*3*2*(w-1)+3] = (x+1)+(y+1)*w;
      tbo_data[x*3*2+y*3*2*(w-1)+4] = (x+1)+(y+0)*w;
      tbo_data[x*3*2+y*3*2*(w-1)+5] = (x+0)+(y+1)*w;
    }
  
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 7 * w * h, vbo_data, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tbo);

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * 3 * 2 * (w-1) * (h-1), tbo_data, GL_STATIC_DRAW);
    
    free(vbo_data);
    free(tbo_data);
    
  }
  
  float sample(glm::vec2 pos) {
  
    int w = data.size();
    int h = data[0].size();
    
    pos.x = (pos.x/hscale) + w/2;
    pos.y = (pos.y/hscale) + h/2;
    
    float a0 = fmod(pos.x, 1.0);
    float a1 = fmod(pos.y, 1.0);
    
    int x0 = (int)std::floor(pos.x), x1 = (int)std::ceil(pos.x);
    int y0 = (int)std::floor(pos.y), y1 = (int)std::ceil(pos.y);
    
    x0 = x0 < 0 ? 0 : x0; x0 = x0 >= w ? w-1 : x0;
    x1 = x1 < 0 ? 0 : x1; x1 = x1 >= w ? w-1 : x1;
    y0 = y0 < 0 ? 0 : y0; y0 = y0 >= h ? h-1 : y0;
    y1 = y1 < 0 ? 0 : y1; y1 = y1 >= h ? h-1 : y1;
    
    float s0 = vscale * (data[x0][y0] - offset);
    float s1 = vscale * (data[x1][y0] - offset);
    float s2 = vscale * (data[x0][y1] - offset);
    float s3 = vscale * (data[x1][y1] - offset);
    
    return (s0 * (1-a0) + s1 * a0) * (1-a1) + (s2 * (1-a0) + s3 * a0) * a1;
  
  }
  
};

static Heightmap* heightmap = NULL;

/* Character */

struct Character {
  // TODO 2: 修改关节数量
  enum { JOINT_NUM = 20 };

  float phase;
  float strafe_amount;
  float strafe_target;
  float crouched_amount;
  float crouched_target;
  float responsive;
  
  glm::vec3 joint_positions[JOINT_NUM];
  glm::vec3 joint_velocities[JOINT_NUM];
  glm::mat3 joint_rotations[JOINT_NUM];
  
  glm::mat4 joint_anim_xform[JOINT_NUM];
  glm::mat4 joint_rest_xform[JOINT_NUM];
  glm::mat4 joint_mesh_xform[JOINT_NUM];
  glm::mat4 joint_global_rest_xform[JOINT_NUM];
  glm::mat4 joint_global_anim_xform[JOINT_NUM];

  int joint_parents[JOINT_NUM];
  
  
  Character()
    , phase(0)
    , strafe_amount(0)
    , strafe_target(0)
    , crouched_amount(0) 
    , crouched_target(0) 
    , responsive(0) {}
    
  void load(const char* filename_v, const char* filename_t, const char* filename_p, const char* filename_r) {
    
    printf("Read Character '%s %s'\n", filename_v, filename_t);
    
    float fparents[JOINT_NUM];
    for (int i = 0; i < JOINT_NUM; i++) { joint_parents[i] = (int)fparents[i]; }
    
    for (int i = 0; i < JOINT_NUM; i++) { joint_rest_xform[i] = glm::transpose(joint_rest_xform[i]); }
  }
  
  void forward_kinematics() {

    for (int i = 0; i < JOINT_NUM; i++) {
      joint_global_anim_xform[i] = joint_anim_xform[i];
      joint_global_rest_xform[i] = joint_rest_xform[i];
      int j = joint_parents[i];
      while (j != -1) {
        joint_global_anim_xform[i] = joint_anim_xform[j] * joint_global_anim_xform[i];
        joint_global_rest_xform[i] = joint_rest_xform[j] * joint_global_rest_xform[i];
        j = joint_parents[j];
      }
      joint_mesh_xform[i] = joint_global_anim_xform[i] * glm::inverse(joint_global_rest_xform[i]);
    }
    
  }
  
};

static Character* character = NULL;

/* Trajectory */

struct Trajectory {
  
  enum { LENGTH = 120 };
  
  float width;

  glm::vec3 positions[LENGTH];
  glm::vec3 directions[LENGTH];
  glm::mat3 rotations[LENGTH];
  float heights[LENGTH];
  
  float gait_stand[LENGTH];
  float gait_walk[LENGTH];
  float gait_jog[LENGTH];
  float gait_crouch[LENGTH];
  float gait_jump[LENGTH];
  float gait_bump[LENGTH];
  
  glm::vec3 target_dir, target_vel;
  
  Trajectory()
    : width(25)
    , target_dir(glm::vec3(0,0,1))
    , target_vel(glm::vec3(0)) {}
  
};

static Trajectory* trajectory = NULL;


/* Areas */

struct Areas {
  
  std::vector<glm::vec3> crouch_pos;
  std::vector<glm::vec2> crouch_size;
  static constexpr float CROUCH_WAVE = 50;
  
  std::vector<glm::vec3> jump_pos;
  std::vector<float> jump_size;
  std::vector<float> jump_falloff;
  
  std::vector<glm::vec2> wall_start;
  std::vector<glm::vec2> wall_stop;
  std::vector<float> wall_width;
  
  void clear() {
    crouch_pos.clear();
    crouch_size.clear();
    jump_pos.clear();
    jump_size.clear();
    jump_falloff.clear();
    wall_start.clear();
    wall_stop.clear();
    wall_width.clear();
  }
  
  void add_wall(glm::vec2 start, glm::vec2 stop, float width) {
    wall_start.push_back(start);
    wall_stop.push_back(stop);
    wall_width.push_back(width);
  }
  
  void add_crouch(glm::vec3 pos, glm::vec2 size) {
    crouch_pos.push_back(pos);
    crouch_size.push_back(size);
  }
  
  void add_jump(glm::vec3 pos, float size, float falloff) {
    jump_pos.push_back(pos);
    jump_size.push_back(size);
    jump_falloff.push_back(falloff);
  }
  
  int num_walls() { return wall_start.size(); }
  int num_crouches() { return crouch_pos.size(); }
  int num_jumps() { return jump_pos.size(); }
  
};

static Areas* areas = NULL;

/* Helper Functions */

static glm::vec3 mix_directions(glm::vec3 x, glm::vec3 y, float a) {
  glm::quat x_q = glm::angleAxis(atan2f(x.x, x.z), glm::vec3(0,1,0));
  glm::quat y_q = glm::angleAxis(atan2f(y.x, y.z), glm::vec3(0,1,0));
  glm::quat z_q = glm::slerp(x_q, y_q, a);
  return z_q * glm::vec3(0,0,1);
}

static glm::mat4 mix_transforms(glm::mat4 x, glm::mat4 y, float a) {
  glm::mat4 out = glm::mat4(glm::slerp(glm::quat(x), glm::quat(y), a));
  out[3] = mix(x[3], y[3], a);
  return out;
}

static glm::quat quat_exp(glm::vec3 l) {
  float w = glm::length(l);
  glm::quat q = w < 0.01 ? glm::quat(1,0,0,0) : glm::quat(
    cosf(w),
    l.x * (sinf(w) / w),
    l.y * (sinf(w) / w),
    l.z * (sinf(w) / w));
  return q / sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z); 
}

static glm::vec2 segment_nearest(glm::vec2 v, glm::vec2 w, glm::vec2 p) {
  float l2 = glm::dot(v - w, v - w);
  if (l2 == 0.0) return v;
  float t = glm::clamp(glm::dot(p - v, w - v) / l2, 0.0f, 1.0f);
  return v + t * (w - v);
}

/* Reset 姿态初始化 */

static void reset(glm::vec2 position) {

  ArrayXf Yp = pfnn->Ymean;
  glm::vec3 root_position = glm::vec3(position.x, heightmap->sample(position), position.y);
  glm::mat3 root_rotation = glm::mat3();

  #pragma region 初始化轨迹
  
  for (int i = 0; i < Trajectory::LENGTH; i++) {
    trajectory->positions[i] = root_position;
    trajectory->rotations[i] = root_rotation;
    trajectory->directions[i] = glm::vec3(0,0,1);
    trajectory->heights[i] = root_position.y;
    trajectory->gait_stand[i] = 0.0;
    trajectory->gait_walk[i] = 0.0;
    trajectory->gait_jog[i] = 0.0;
    trajectory->gait_crouch[i] = 0.0;
    trajectory->gait_jump[i] = 0.0;
    trajectory->gait_bump[i] = 0.0;
  }

  #pragma endregion
  

  #pragma region 初始化角色

  for (int i = 0; i < Character::JOINT_NUM; i++) {
    
    int opos = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*0);
    int ovel = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*1);
    int orot = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*2);
    
    glm::vec3 pos = (root_rotation * glm::vec3(Yp(opos+i*3+0), Yp(opos+i*3+1), Yp(opos+i*3+2))) + root_position;
    glm::vec3 vel = (root_rotation * glm::vec3(Yp(ovel+i*3+0), Yp(ovel+i*3+1), Yp(ovel+i*3+2)));
    glm::mat3 rot = (root_rotation * glm::toMat3(quat_exp(glm::vec3(Yp(orot+i*3+0), Yp(orot+i*3+1), Yp(orot+i*3+2)))));
    
    character->joint_positions[i]  = pos;
    character->joint_velocities[i] = vel;
    character->joint_rotations[i]  = rot;
  }
  
  character->phase = 0.0;

  #pragma endregion
}


static void pre_render() {
        
  #pragma region 处理手柄输入

  /* Update Camera */
  // 鼠标滚轮拉近和拉远摄像头
  int y_move = SDL_JoystickGetAxis(stick, GAMEPAD_STICK_R_HORIZONTAL);
  int x_move = SDL_JoystickGetAxis(stick, GAMEPAD_STICK_R_VERTICAL);
  x_move = -x_move;
  y_move = 0;
  
  if (abs(x_move) + abs(y_move) < 10000) { x_move = 0; y_move = 0; };
  if (options->invert_y) { y_move = -y_move; }
  camera->pitch = glm::clamp(camera->pitch + (y_move / 32768.0) * 0.03, M_PI/16, 2*M_PI/5);
  camera->yaw = camera->yaw + (x_move / 32768.0) * 0.03;
  float zoom_i = SDL_JoystickGetButton(stick, GAMEPAD_SHOULDER_L) * 20.0;
  float zoom_o = SDL_JoystickGetButton(stick, GAMEPAD_SHOULDER_R) * 20.0;
  if (zoom_i > 1e-5) { camera->distance = glm::clamp(camera->distance + zoom_i, 10.0f, 10000.0f); }
  if (zoom_o > 1e-5) { camera->distance = glm::clamp(camera->distance - zoom_o, 10.0f, 10000.0f); }

  // 左右移动
  int x_vel = -SDL_JoystickGetAxis(stick, GAMEPAD_STICK_L_HORIZONTAL);
  // 前后移动
  int y_vel = -SDL_JoystickGetAxis(stick, GAMEPAD_STICK_L_VERTICAL); 
  if (abs(x_vel) + abs(y_vel) < 10000) { x_vel = 0; y_vel = 0; };  

  #pragma endregion


  #pragma region 计算&更新轨迹的目标方向和速度 "trajectory->target_vel 应用到 trajectory_positions_blend", "trajectory->target_dir 应用到 trajectory->directions"

  /* Update Target Direction / Velocity */
  /* 更新目标方向和速度 */

  // 1a 相机位置
  glm::vec3 trajectory_target_dir_new = glm::normalize(glm::vec3(camera->direction().x, 0.0, camera->direction().z));
  // 1b 相机方向, 后面人体朝向会在轨迹方向和相机方向范围内偏转
  glm::mat3 trajectory_target_rotation = glm::mat3(glm::rotate(atan2f(
    trajectory_target_dir_new.x,
    trajectory_target_dir_new.z), glm::vec3(0,1,0)));
  

  // 2a 目标速度大小 = 手柄输入
  float target_vel_speed = 2.5 + 2.5 * ((SDL_JoystickGetAxis(stick, GAMEPAD_TRIGGER_R) / 32768.0) + 1.0);
  // 2b 目标速度向量 = 目标速度大小 * 相机视角朝向 * 手柄输入
  glm::vec3 trajectory_target_vel_new = target_vel_speed * (trajectory_target_rotation * glm::vec3(x_vel / 32768.0, 0, y_vel / 32768.0));
  // 2c 平滑过渡到目标速度向量
  trajectory->target_vel = glm::mix(trajectory->target_vel, trajectory_target_vel_new, options->extra_velocity_smooth);
  

  // 3a 面向面朝方向 = 手柄输入
  character->strafe_target = ((SDL_JoystickGetAxis(stick, GAMEPAD_TRIGGER_L) / 32768.0) + 1.0) / 2.0;
  // 3b strafe_amount控制面向的偏转程度, 平滑过度
  character->strafe_amount = glm::mix(character->strafe_amount, character->strafe_target, options->extra_strafe_smooth);
  

  // 4a 如果目标速度太小 = 认为目标方向是当前方向, 即不变
  // 4b 如果目标速度正常大小 = 取目标速度向量的单位向量作为目标方向
  glm::vec3 trajectory_target_vel_dir = glm::length(trajectory->target_vel) < 1e-05 ? trajectory->target_dir : glm::normalize(trajectory->target_vel);
  // 4c 在轨迹方向和相机方向之间混合
  trajectory_target_dir_new = mix_directions(trajectory_target_vel_dir, trajectory_target_dir_new, character->strafe_amount);
  // 4d 轨迹最终方向 = 上一帧 平滑至 混合后的方向
  trajectory->target_dir = mix_directions(trajectory->target_dir, trajectory_target_dir_new, options->extra_direction_smooth);  
  
  #pragma endregion


  #pragma region 更新轨迹的步态

  /* Update Gait */
  
  if (glm::length(trajectory->target_vel) < 0.1)  {
    // 速度太慢就站立
    float stand_amount = 1.0f - glm::clamp(glm::length(trajectory->target_vel) / 0.1f, 0.0f, 1.0f);
    trajectory->gait_stand[Trajectory::LENGTH/2]  = glm::mix(trajectory->gait_stand[Trajectory::LENGTH/2],  stand_amount, options->extra_gait_smooth);
    trajectory->gait_walk[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_walk[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
    trajectory->gait_jog[Trajectory::LENGTH/2]    = glm::mix(trajectory->gait_jog[Trajectory::LENGTH/2],    0.0f, options->extra_gait_smooth);
    trajectory->gait_crouch[Trajectory::LENGTH/2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH/2], 0.0f, options->extra_gait_smooth);
    trajectory->gait_jump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_jump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
    trajectory->gait_bump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_bump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
  // } else if ((SDL_JoystickGetAxis(stick, GAMEPAD_TRIGGER_R) / 32768.0) + 1.0) {
  } else if (false) {
    // 慢跑
    trajectory->gait_stand[Trajectory::LENGTH/2]  = glm::mix(trajectory->gait_stand[Trajectory::LENGTH/2],  0.0f, options->extra_gait_smooth);
    trajectory->gait_walk[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_walk[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);
    trajectory->gait_jog[Trajectory::LENGTH/2]    = glm::mix(trajectory->gait_jog[Trajectory::LENGTH/2],    1.0f, options->extra_gait_smooth);
    trajectory->gait_crouch[Trajectory::LENGTH/2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH/2], 0.0f, options->extra_gait_smooth);
    trajectory->gait_jump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_jump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);    
    trajectory->gait_bump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_bump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);    
  } else {
    // 走
    trajectory->gait_stand[Trajectory::LENGTH/2]  = glm::mix(trajectory->gait_stand[Trajectory::LENGTH/2],  0.0f, options->extra_gait_smooth);
    trajectory->gait_walk[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_walk[Trajectory::LENGTH/2],   1.0f, options->extra_gait_smooth);
    trajectory->gait_jog[Trajectory::LENGTH/2]    = glm::mix(trajectory->gait_jog[Trajectory::LENGTH/2],    0.0f, options->extra_gait_smooth);
    trajectory->gait_crouch[Trajectory::LENGTH/2] = glm::mix(trajectory->gait_crouch[Trajectory::LENGTH/2], 0.0f, options->extra_gait_smooth);
    trajectory->gait_jump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_jump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);  
    trajectory->gait_bump[Trajectory::LENGTH/2]   = glm::mix(trajectory->gait_bump[Trajectory::LENGTH/2],   0.0f, options->extra_gait_smooth);  
  }

  #pragma endregion


  #pragma region 预测轨迹的位置,方向,高度,步态 trajectory->positions,directions,heights,gaits

  /* Predict Future Trajectory */
  // 根据手柄的控制, 不断更新将来帧的可能轨迹位置和方向
  glm::vec3 trajectory_positions_blend[Trajectory::LENGTH];
  trajectory_positions_blend[Trajectory::LENGTH/2] = trajectory->positions[Trajectory::LENGTH/2];

  // 本质上是轨迹上的每个点都会有个 scale factor, 越远影响越大
  for (int i = Trajectory::LENGTH/2+1; i < Trajectory::LENGTH; i++) {
    // 更新61帧以后帧的方向高度和步态
    float bias_pos = character->responsive ? glm::mix(2.0f, 2.0f, character->strafe_amount) : glm::mix(0.5f, 1.0f, character->strafe_amount);
    float bias_dir = character->responsive ? glm::mix(5.0f, 3.0f, character->strafe_amount) : glm::mix(2.0f, 0.5f, character->strafe_amount);
    
    float scale_pos = (1.0f - powf(1.0f - ((float)(i - Trajectory::LENGTH/2) / (Trajectory::LENGTH/2)), bias_pos));
    float scale_dir = (1.0f - powf(1.0f - ((float)(i - Trajectory::LENGTH/2) / (Trajectory::LENGTH/2)), bias_dir));

    // 根据移动速度更新未来帧的位置
    trajectory_positions_blend[i] = trajectory_positions_blend[i-1] + glm::mix(
        trajectory->positions[i] - trajectory->positions[i-1], 
        trajectory->target_vel,
        scale_pos);

    // 人体方向：当前方向朝目标方向慢慢过渡
    trajectory->directions[i] = mix_directions(trajectory->directions[i], trajectory->target_dir, scale_dir);
    
    /* 未来61帧的高度、Gait和当前帧一样 */
    // 轨迹地形高度, 将来的地形高度都是当前时刻中间帧(第60帧)的轨迹高度
    trajectory->heights[i] = trajectory->heights[Trajectory::LENGTH/2]; 
    // 步态
    trajectory->gait_stand[i]  = trajectory->gait_stand[Trajectory::LENGTH/2]; 
    trajectory->gait_walk[i]   = trajectory->gait_walk[Trajectory::LENGTH/2];  
    trajectory->gait_jog[i]    = trajectory->gait_jog[Trajectory::LENGTH/2];   
    trajectory->gait_crouch[i] = trajectory->gait_crouch[Trajectory::LENGTH/2];
    trajectory->gait_jump[i]   = trajectory->gait_jump[Trajectory::LENGTH/2];  
    trajectory->gait_bump[i]   = trajectory->gait_bump[Trajectory::LENGTH/2];  
  }
  
  for (int i = Trajectory::LENGTH/2+1; i < Trajectory::LENGTH; i++) {
    trajectory->positions[i] = trajectory_positions_blend[i];
  }

  /* Trajectory Rotation */
  for (int i = 0; i < Trajectory::LENGTH; i++) {
    trajectory->rotations[i] = glm::mat3(glm::rotate(atan2f(
      trajectory->directions[i].x,
      trajectory->directions[i].z), glm::vec3(0,1,0)));
  }

  /* Trajectory Heights */
  for (int i = Trajectory::LENGTH/2; i < Trajectory::LENGTH; i++) {
    // 地形图的当前位置高度
    trajectory->positions[i].y = heightmap->sample(glm::vec2(trajectory->positions[i].x, trajectory->positions[i].z));
  }
    
  trajectory->heights[Trajectory::LENGTH/2] = 0.0;
  for (int i = 0; i < Trajectory::LENGTH; i+=10) {
    // 地形高度的平均
    trajectory->heights[Trajectory::LENGTH/2] += (trajectory->positions[i].y / ((Trajectory::LENGTH)/10));
  }

  #pragma endregion


  #pragma region 准备网络输入
  
  // 人体当前帧的位置和高度
  glm::vec3 root_position = glm::vec3(
    trajectory->positions[Trajectory::LENGTH/2].x, 
    trajectory->heights[Trajectory::LENGTH/2],
    trajectory->positions[Trajectory::LENGTH/2].z);
          
  // 人体当前帧的根关节方向
  glm::mat3 root_rotation = trajectory->rotations[Trajectory::LENGTH/2];


  /* Input Trajectory Positions / Directions */
  for (int i = 0; i < Trajectory::LENGTH; i+=10) {
    int w = (Trajectory::LENGTH)/10;
    glm::vec3 pos = glm::inverse(root_rotation) * (trajectory->positions[i] - root_position);
    glm::vec3 dir = glm::inverse(root_rotation) * trajectory->directions[i];  
    pfnn->Xp((w*0)+i/10) = pos.x;
    pfnn->Xp((w*1)+i/10) = pos.z;
    pfnn->Xp((w*2)+i/10) = dir.x;
    pfnn->Xp((w*3)+i/10) = dir.z;
  }
    
  /* Input Trajectory Gaits */
  for (int i = 0; i < Trajectory::LENGTH; i+=10) {
    int w = (Trajectory::LENGTH)/10;
    pfnn->Xp((w*4)+i/10) = trajectory->gait_stand[i];
    pfnn->Xp((w*5)+i/10) = trajectory->gait_walk[i];
    pfnn->Xp((w*6)+i/10) = trajectory->gait_jog[i];
    pfnn->Xp((w*7)+i/10) = trajectory->gait_crouch[i];
    pfnn->Xp((w*8)+i/10) = trajectory->gait_jump[i];
    pfnn->Xp((w*9)+i/10) = 0.0; // Unused.
  }

  /* Input Joint Previous Positions / Velocities / Rotations */
  glm::vec3 prev_root_position = glm::vec3(
    trajectory->positions[Trajectory::LENGTH/2-1].x, 
    trajectory->heights[Trajectory::LENGTH/2-1],
    trajectory->positions[Trajectory::LENGTH/2-1].z);
   
  glm::mat3 prev_root_rotation = trajectory->rotations[Trajectory::LENGTH/2-1];
  
  for (int i = 0; i < Character::JOINT_NUM; i++) {
    int o = (((Trajectory::LENGTH)/10)*10);  
    glm::vec3 pos = glm::inverse(prev_root_rotation) * (character->joint_positions[i] - prev_root_position);
    glm::vec3 prv = glm::inverse(prev_root_rotation) *  character->joint_velocities[i];
    pfnn->Xp(o+(Character::JOINT_NUM*3*0)+i*3+0) = pos.x;
    pfnn->Xp(o+(Character::JOINT_NUM*3*0)+i*3+1) = pos.y;
    pfnn->Xp(o+(Character::JOINT_NUM*3*0)+i*3+2) = pos.z;
    pfnn->Xp(o+(Character::JOINT_NUM*3*1)+i*3+0) = prv.x;
    pfnn->Xp(o+(Character::JOINT_NUM*3*1)+i*3+1) = prv.y;
    pfnn->Xp(o+(Character::JOINT_NUM*3*1)+i*3+2) = prv.z;
  }
    
  /* Input Trajectory Heights */
  for (int i = 0; i < Trajectory::LENGTH; i += 10) {
    int o = (((Trajectory::LENGTH)/10)*10)+Character::JOINT_NUM*3*2;
    int w = (Trajectory::LENGTH)/10;
    glm::vec3 position_r = trajectory->positions[i] + (trajectory->rotations[i] * glm::vec3( trajectory->width, 0, 0));
    glm::vec3 position_l = trajectory->positions[i] + (trajectory->rotations[i] * glm::vec3(-trajectory->width, 0, 0));
    pfnn->Xp(o+(w*0)+(i/10)) = heightmap->sample(glm::vec2(position_r.x, position_r.z)) - root_position.y;
    pfnn->Xp(o+(w*1)+(i/10)) = trajectory->positions[i].y - root_position.y;
    pfnn->Xp(o+(w*2)+(i/10)) = heightmap->sample(glm::vec2(position_l.x, position_l.z)) - root_position.y;
  }

  #pragma endregion
    

  /* Perform Regression 网络预测 */
  pfnn->predict(character->phase);

    
  #pragma region 处理网络输出,应用到关节的局部位置,速度,旋转 -> 通过前向动力学转换到全局
  /* Build Local Transforms */
  
  for (int i = 0; i < Character::JOINT_NUM; i++) {
    int opos = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*0);
    int ovel = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*1);
    int orot = 8+(((Trajectory::LENGTH/2)/10)*4)+(Character::JOINT_NUM*3*2);
    
    glm::vec3 pos = (root_rotation * glm::vec3(pfnn->Yp(opos+i*3+0), pfnn->Yp(opos+i*3+1), pfnn->Yp(opos+i*3+2))) + root_position;
    glm::vec3 vel = (root_rotation * glm::vec3(pfnn->Yp(ovel+i*3+0), pfnn->Yp(ovel+i*3+1), pfnn->Yp(ovel+i*3+2)));
    glm::mat3 rot = (root_rotation * glm::toMat3(quat_exp(glm::vec3(pfnn->Yp(orot+i*3+0), pfnn->Yp(orot+i*3+1), pfnn->Yp(orot+i*3+2)))));
    
    /*
    ** Blending Between the predicted positions and
    ** the previous positions plus the velocities 
    ** smooths out the motion a bit in the case 
    ** where the two disagree with each other.
    */
    
    character->joint_positions[i]  = glm::mix(character->joint_positions[i] + vel, pos, options->extra_joint_smooth);
    character->joint_velocities[i] = vel;
    character->joint_rotations[i]  = rot;
  }
  
  /* Convert to local space ... yes I know this is inefficient. */
  
  for (int i = 0; i < Character::JOINT_NUM; i++) {
    if (i == 0) {
      character->joint_anim_xform[i] = character->joint_global_anim_xform[i];
    } else {
      character->joint_anim_xform[i] = glm::inverse(character->joint_global_anim_xform[character->joint_parents[i]]) * character->joint_global_anim_xform[i];
    }
  }

  character->forward_kinematics();

  #pragma endregion

}

void post_render() {
            
  /* Update Past Trajectory */
  // 1~60窗口向右移动1帧
  for (int i = 0; i < Trajectory::LENGTH/2; i++) {
    trajectory->positions[i]  = trajectory->positions[i+1];
    trajectory->directions[i] = trajectory->directions[i+1];
    trajectory->rotations[i] = trajectory->rotations[i+1];
    trajectory->heights[i] = trajectory->heights[i+1];
    trajectory->gait_stand[i] = trajectory->gait_stand[i+1];
    trajectory->gait_walk[i] = trajectory->gait_walk[i+1];
    trajectory->gait_jog[i] = trajectory->gait_jog[i+1];
    trajectory->gait_crouch[i] = trajectory->gait_crouch[i+1];
    trajectory->gait_jump[i] = trajectory->gait_jump[i+1];
    trajectory->gait_bump[i] = trajectory->gait_bump[i+1];
  }
  
  /* Update Current Trajectory */
  // 第61帧(下标60), 结合pfnn的输出, 更新position,direction,rotation
  // 也就是当前位置
  float stand_amount = powf(1.0f-trajectory->gait_stand[Trajectory::LENGTH/2], 0.25f);
  
  glm::vec3 trajectory_update = (trajectory->rotations[Trajectory::LENGTH/2] * glm::vec3(pfnn->Yp(0), 0, pfnn->Yp(1)));
  trajectory->positions[Trajectory::LENGTH/2]  = trajectory->positions[Trajectory::LENGTH/2] + stand_amount * trajectory_update;
  trajectory->directions[Trajectory::LENGTH/2] = glm::mat3(glm::rotate(stand_amount * -pfnn->Yp(2), glm::vec3(0,1,0))) * trajectory->directions[Trajectory::LENGTH/2];
  trajectory->rotations[Trajectory::LENGTH/2] = glm::mat3(glm::rotate(atan2f(
      trajectory->directions[Trajectory::LENGTH/2].x,
      trajectory->directions[Trajectory::LENGTH/2].z), glm::vec3(0,1,0)));


  /* Update Future Trajectory */
  // 第62~120帧, 结合pfnn的输出, 更新position,direction,rotation
  for (int i = Trajectory::LENGTH/2+1; i < Trajectory::LENGTH; i++) {
    int w = (Trajectory::LENGTH/2)/10;
    float m = fmod(((float)i - (Trajectory::LENGTH/2)) / 10.0, 1.0);
    trajectory->positions[i].x  = (1-m) * pfnn->Yp(8+(w*0)+(i/10)-w) + m * pfnn->Yp(8+(w*0)+(i/10)-w+1);
    trajectory->positions[i].z  = (1-m) * pfnn->Yp(8+(w*1)+(i/10)-w) + m * pfnn->Yp(8+(w*1)+(i/10)-w+1);
    trajectory->directions[i].x = (1-m) * pfnn->Yp(8+(w*2)+(i/10)-w) + m * pfnn->Yp(8+(w*2)+(i/10)-w+1);
    trajectory->directions[i].z = (1-m) * pfnn->Yp(8+(w*3)+(i/10)-w) + m * pfnn->Yp(8+(w*3)+(i/10)-w+1);
    trajectory->positions[i]    = (trajectory->rotations[Trajectory::LENGTH/2] * trajectory->positions[i]) + trajectory->positions[Trajectory::LENGTH/2];
    trajectory->directions[i]   = glm::normalize((trajectory->rotations[Trajectory::LENGTH/2] * trajectory->directions[i]));
    trajectory->rotations[i]    = glm::mat3(glm::rotate(atan2f(trajectory->directions[i].x, trajectory->directions[i].z), glm::vec3(0,1,0)));
  }
  
  /* Update Phase */
  // 更新相位
  character->phase = fmod(character->phase + (stand_amount * 0.9f + 0.1f) * 2*M_PI * pfnn->Yp(3), 2*M_PI);
}

void render() {
  /* Render Trajectory */
  
  if (options->display_debug) {
    glPointSize(1.0 * options->display_scale);
    glBegin(GL_POINTS);
    for (int i = 0; i < Trajectory::LENGTH-10; i++) {
      glm::vec3 position_c = trajectory->positions[i];
      glColor3f(trajectory->gait_jump[i], trajectory->gait_bump[i], trajectory->gait_crouch[i]);
      glVertex3f(position_c.x, position_c.y + 2.0, position_c.z);
    }
    glEnd();
    glColor3f(1.0, 1.0, 1.0);
    glPointSize(1.0);

    
    glPointSize(4.0 * options->display_scale);
    glBegin(GL_POINTS);
    for (int i = 0; i < Trajectory::LENGTH; i+=10) {
      glm::vec3 position_c = trajectory->positions[i];
      glColor3f(trajectory->gait_jump[i], trajectory->gait_bump[i], trajectory->gait_crouch[i]);
      glVertex3f(position_c.x, position_c.y + 2.0, position_c.z);
    }
    glEnd();
    glColor3f(1.0, 1.0, 1.0);
    glPointSize(1.0);


    glLineWidth(1.0 * options->display_scale);
    glBegin(GL_LINES);
    for (int i = 0; i < Trajectory::LENGTH; i+=10) {
      glm::vec3 base = trajectory->positions[i] + glm::vec3(0.0, 2.0, 0.0);
      glm::vec3 side = glm::normalize(glm::cross(trajectory->directions[i], glm::vec3(0.0, 1.0, 0.0)));
      glm::vec3 fwrd = base + 15.0f * trajectory->directions[i];
      fwrd.y = heightmap->sample(glm::vec2(fwrd.x, fwrd.z)) + 2.0;
      glm::vec3 arw0 = fwrd +  4.0f * side + 4.0f * -trajectory->directions[i];
      glm::vec3 arw1 = fwrd -  4.0f * side + 4.0f * -trajectory->directions[i];
      glColor3f(trajectory->gait_jump[i], trajectory->gait_bump[i], trajectory->gait_crouch[i]);
      glVertex3f(base.x, base.y, base.z);
      glVertex3f(fwrd.x, fwrd.y, fwrd.z);
      glVertex3f(fwrd.x, fwrd.y, fwrd.z);
      glVertex3f(arw0.x, fwrd.y, arw0.z);
      glVertex3f(fwrd.x, fwrd.y, fwrd.z);
      glVertex3f(arw1.x, fwrd.y, arw1.z);
    }
    glEnd();
    glLineWidth(1.0);
    glColor3f(1.0, 1.0, 1.0);
  }
  
  /* Render Joints */
  
  // if (options->display_debug && options->display_debug_joints) {
  if (true) {
    for (int i = 0; i < Character::JOINT_NUM; i++) {
      glm::vec3 pos = character->joint_positions[i];
    }
    for (int i = 0; i < Character::JOINT_NUM; i++) {
      glm::vec3 pos = character->joint_positions[i];
      glm::vec3 vel = pos - 5.0f * character->joint_velocities[i];
    }
  }
  
  /* PFNN Visual */
  
  if (options->display_debug && options->display_debug_pfnn) {
    
    glColor3f(0.0, 0.0, 0.0);  

    glLineWidth(5);
    glBegin(GL_LINES);
    for (float i = 0; i < 2*M_PI; i+=0.01) {
      glVertex3f(WINDOW_WIDTH-125+50*cos(i     ),100+50*sin(i     ),0);    
      glVertex3f(WINDOW_WIDTH-125+50*cos(i+0.01),100+50*sin(i+0.01),0);
    }
    glEnd();
    glLineWidth(1);
    
    glPointSize(20);
    glBegin(GL_POINTS);
    glVertex3f(WINDOW_WIDTH-125+50*cos(character->phase), 100+50*sin(character->phase), 0);
    glEnd();
    glPointSize(1);
    
    glColor3f(1.0, 1.0, 1.0); 

    int pindex_1 = (int)((character->phase / (2*M_PI)) * 50);
    MatrixXf W0p = pfnn->W0[pindex_1];

    glPointSize(1);
    glBegin(GL_POINTS);
    
    for (int x = 0; x < W0p.rows(); x++)
    for (int y = 0; y < W0p.cols(); y++) {
      float v = (W0p(x, y)+0.5)/2.0;
      glm::vec3 col = v > 0.5 ? glm::mix(glm::vec3(1,0,0), glm::vec3(0,0,1), v-0.5) : glm::mix(glm::vec3(0,1,0), glm::vec3(0,0,1), v*2.0);
      glColor3f(col.x, col.y, col.z); 
      glVertex3f(WINDOW_WIDTH-W0p.cols()+y-25, x+175, 0);
    }
    
    glEnd();
    glPointSize(1);
    
  }
}

int main(int argc, char **argv) {
  
  stick = SDL_JoystickOpen(0);
  
  /* Resources */
  character = new Character();
  character->load(
    "./network/character_vertices.bin", 
    "./network/character_triangles.bin", 
    "./network/character_parents.bin", 
    "./network/character_xforms.bin");
  
  trajectory = new Trajectory();
  heightmap = new Heightmap();
  pfnn = new PFNN(PFNN::MODE_CONSTANT);
  pfnn->load();


  /* Load World */
  heightmap->load("./heightmaps/hmap_000_smooth.txt", 1.0);


  /* Init Character and Trajectory, Pos Rot Vel ... */
  reset(glm::vec2(0, 0));


  /* Game Loop */
  
  static bool running = true;
  
  while (running) {
    pre_render();
    render();
    post_render();
  }
}