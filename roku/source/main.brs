sub Main()
    screen = CreateObject("roSGScreen")
    port = CreateObject("roMessagePort")
    screen.SetMessagePort(port)
    screen.CreateScene("WindowScene")
    screen.Show()
    ' The window runs all day without anyone touching the remote. Roku closes apps after a couple
    ' of hours with no key presses (EXIT_IDLE_AUTO_EXIT), so tell it someone is here every 5 minutes.
    apps = CreateObject("roAppManager")
    while true
        msg = wait(5 * 60 * 1000, port)
        if msg = invalid
            apps.UpdateLastKeyPressTime()
        else if type(msg) = "roSGScreenEvent" and msg.IsScreenClosed()
            return
        end if
    end while
end sub
