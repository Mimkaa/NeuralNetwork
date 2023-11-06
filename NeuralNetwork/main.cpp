#include <memory.h>
#include "Game.h"

int main()
{
    std::unique_ptr<Game> game = std::make_unique<Game>(800, 800, "NN");

    game->run(60);

    return 0;
}